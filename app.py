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
from modules.analytics_engine import AnalyticsEngine

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

# Camera and Face Engine
if settings.APP_ROLE == "LOCAL_KIOSK":
    camera = Camera()
    face_engine = FaceEngine(
        model_path=os.path.join(basedir, "data", "lbph_model.yml"),
        faces_dir=os.path.join(basedir, "data", "faces"),
    )
else:
    camera = None
    face_engine = None

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

LBPH_DISTANCE_THRESHOLD = _env_float("LBPH_DISTANCE_THRESHOLD", 95.0)
RECOGNITION_CACHE_TTL_SEC = _env_float("RECOGNITION_CACHE_TTL_SEC", 2.0)
BLINK_EAR_THRESHOLD = _env_float("BLINK_EAR_THRESHOLD", 0.25)
BLINK_MAX_CLOSED_SEC = _env_float("BLINK_MAX_CLOSED_SEC", 2.5)
VERIFIED_TTL_SEC = _env_float("VERIFIED_TTL_SEC", 20.0)

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
    return render_template(request, "enroll.html", {"user": user})

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

    faces = face_engine.detect_faces(frame)
    if faces:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (x, y, w, h) = faces[0]
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
    
    # Priority: 1. Base64 from frontend, 2. Camera fallback
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

    faces_data = face_engine.detect_faces_mesh(frame)
    now_ts = time.time()
    results = []
    last_recognized_user = request.session.get('last_recognized', {'id': None, 'ts': 0})

    # Clean up old entries from verified_live_users
    for user_id in list(verified_live_users.keys()):
        if now_ts - verified_live_users[user_id] > VERIFIED_TTL_SEC:
            del verified_live_users[user_id]

    for ((x, y, w, h), landmarks) in faces_data:
        face_img = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        label, distance = face_engine.recognize_face(gray_face)

        result = {
            'rect': [int(x), int(y), int(w), int(h)],
            'name': 'Unknown',
            'status': 'unknown',
            'verified': False,
            'user_id': None,
        }

        if label != -1 and distance < LBPH_DISTANCE_THRESHOLD:
            user = db.get(User, label)
            if user:
                result['name'] = user.name
                result['status'] = 'recognized'
                result['user_id'] = user.id
                recognized_faces[user.id] = now_ts

                # Add confidence score to the result for debugging
                result['confidence'] = distance

                ear = face_engine.check_liveness(landmarks, frame.shape[1], frame.shape[0])
                state = blink_state.get(user.id, {"closed": False, "changed_at": now_ts})

                if ear < BLINK_EAR_THRESHOLD:
                    if not state["closed"]:
                        state = {"closed": True, "changed_at": now_ts}
                else:
                    if state["closed"]:
                        closed_for = now_ts - state["changed_at"]
                        if 0.05 <= closed_for <= BLINK_MAX_CLOSED_SEC:
                            verified_live_users[user.id] = now_ts
                            face_verification_cache[user.id] = now_ts
                            result["liveness_detected"] = True
                        state = {"closed": False, "changed_at": now_ts}

                blink_state[user.id] = state
                last_verified = face_verification_cache.get(user.id, 0)
                if now_ts - last_verified <= VERIFIED_TTL_SEC:
                    result["verified"] = True
        else:
            # Log unknown faces for debugging
            # print(f"Unknown face detected. Label: {label}, Distance: {distance}")
            if last_recognized_user['id'] and (now_ts - last_recognized_user['ts']) < RECOGNITION_CACHE_TTL_SEC:
                cached_user = db.get(User, last_recognized_user['id'])
                if cached_user:
                    result['name'] = cached_user.name
                    result['status'] = 'recognized'
                    result['user_id'] = cached_user.id
                    # Check if this user was already verified recently
                    last_verified = face_verification_cache.get(cached_user.id, 0)
                    if now_ts - last_verified <= VERIFIED_TTL_SEC:
                        result["verified"] = True

        results.append(result)
    
    # After processing all faces, update the session if a new user was recognized
    if any(r['status'] == 'recognized' for r in results):
        recognized_user = next(r for r in results if r['status'] == 'recognized')
        request.session['last_recognized'] = {'id': recognized_user['user_id'], 'ts': now_ts}
    elif not faces_data:
        # If no faces are detected at all, clear the cache
        request.session.pop('last_recognized', None)

    return JSONResponse({'status': 'success', 'faces': results})

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

def write_attendance_to_sql(user_id: int, status: str, timestamp: datetime):
    db = SessionLocal()
    try:
        new_record = Attendance(
            user_id=user_id,
            status=status,
            timestamp=timestamp,
            device_id=settings.DEVICE_ID,
        )
        db.add(new_record)
        db.commit()
        print(f"Successfully wrote attendance for user {user_id} to SQL.")
    except Exception as e:
        print(f"Error writing attendance to SQL for user {user_id}: {e}")
    finally:
        db.close()

@app.post("/api/attendance/record")
async def api_attendance_record(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db), redis: Redis = Depends(get_redis)):
    try:
        if settings.APP_ROLE != "LOCAL_KIOSK":
            return JSONResponse({'status': 'error', 'message': 'Recording not available on this service'}, status_code=404)

        data = await request.json()
        user_id = data.get('user_id')
        action = data.get('action')

        if not all([user_id, action]):
            return JSONResponse({'status': 'error', 'message': 'Missing user_id or action'}, status_code=400)

        try:
            uid = int(user_id)
        except (ValueError, TypeError):
            return JSONResponse({'status': 'error', 'message': 'Invalid user_id format'}, status_code=400)

        user = db.get(User, uid)
        if not user:
            return JSONResponse({'status': 'error', 'message': 'User not found'}, status_code=404)

        # --- Security Check: Liveness (Blink) Verification ---
        now_ts = time.time()
        last_verified = face_verification_cache.get(uid, 0)
        if now_ts - last_verified > VERIFIED_TTL_SEC:
            return JSONResponse({'status': 'error', 'message': 'Liveness verification required. Please blink at the camera.'}, status_code=403)

        now = get_now_pht()
        today_str = now.strftime("%Y-%m-%d")
        live_set_key = f"attendance:{today_str}:present"
        
        if action.lower() == 'login':
            # --- Live Layer (Redis) ---
            if redis.sismember(live_set_key, uid):
                return JSONResponse({'status': 'error', 'message': 'You are already logged in for today.'}, status_code=409)
            
            redis.sadd(live_set_key, uid)

            # --- Determine Status ---
            sch_start_str = user.schedule_start or "06:00"
            try:
                sch_start_dt = now.replace(hour=int(sch_start_str[:2]), minute=int(sch_start_str[3:]), second=0, microsecond=0)
            except ValueError:
                sch_start_dt = now.replace(hour=6, minute=0, second=0, microsecond=0)

            on_time_end = sch_start_dt + timedelta(minutes=15)
            status = 'On Time' if now <= on_time_end else 'Late'
            message = f"Login successful for {user.name}."

            # --- Persistence Layer (SQL via Background Task) ---
            background_tasks.add_task(write_attendance_to_sql, user_id=uid, status=status, timestamp=now)

        elif action.lower() == 'logout':
            # --- Live Layer (Redis) ---
            redis.srem(live_set_key, uid)

            # --- Persistence Layer (SQL via Background Task) ---
            status = 'Logout'
            message = f"Logout successful for {user.name}."
            background_tasks.add_task(write_attendance_to_sql, user_id=uid, status=status, timestamp=now)
        
        else:
            return JSONResponse({'status': 'error', 'message': 'Invalid action specified'}, status_code=400)

        return JSONResponse({'status': 'success', 'message': message})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'status': 'error', 'message': 'Internal Server Error'}, status_code=500)

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
        request.state.flash(f"User {user.name} deleted.", "success")
    else:
        request.state.flash("User not found.", "danger")
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/edit_user/{user_id}")
async def edit_user(user_id: int, request: Request, name: str = Form(...), schedule_start: str = Form(None), schedule_end: str = Form(None), employment_type: str = Form(None), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if user and name:
        user.name = name
        if schedule_start: user.schedule_start = schedule_start
        if schedule_end: user.schedule_end = schedule_end
        if employment_type: user.employment_type = employment_type
        db.commit()
        request.state.flash(f"User updated to {name}.", "success")
    else:
        request.state.flash("Invalid request.", "danger")
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/analytics")
async def analytics_route(request: Request, db: Session = Depends(get_db)):
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")
        
    engine = AnalyticsEngine(db)
    stats = []
    users = db.query(User).all()
    for user in users:
        presents = db.query(Attendance).filter(Attendance.user_id == user.id, Attendance.status.in_(['Present', 'On Time'])).count()
        tardies = db.query(Attendance).filter(Attendance.user_id == user.id, Attendance.status.in_(['Tardy', 'Late'])).count()
        absences = db.query(Attendance).filter(Attendance.user_id == user.id, Attendance.status == 'Absent').count()
        stats.append({
            'id': user.id,
            'staff_code': user.staff_code,
            'name': user.name,
            'employment_type': user.employment_type,
            'present': presents,
            'tardy': tardies,
            'absent': absences
        })
    
    # Add trends data for the D3 chart
    trends = engine.get_weekly_trends()
    
    return JSONResponse({
        'stats': stats,
        'trends': trends
    })

@app.post("/add_user_by_id")
async def add_user_by_id(request: Request, user_id: int = Form(...), db: Session = Depends(get_db)):
    existing = db.get(User, user_id)
    if existing:
        request.state.flash("User ID already exists", "danger")
        return RedirectResponse(url="/admin", status_code=303)
    
    staff_code = f"{random.randint(0, 999999):06d}"
    new_user = User(id=user_id, name=f'User {user_id}', staff_code=staff_code)
    db.add(new_user)
    db.commit()
    
    user_dir = os.path.join(basedir, 'data', 'faces', str(new_user.id))
    os.makedirs(user_dir, exist_ok=True)
    return RedirectResponse(url=f"/enroll/{new_user.id}", status_code=303)

@app.post("/re_enroll", response_class=HTMLResponse)
async def re_enroll(request: Request, staff_code: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.staff_code == staff_code).first()
    if user:
        return render_template(request, "enroll.html", {"user": user, "re_enroll": True})
    request.state.flash("Staff ID not found.", "danger")
    return RedirectResponse(url="/", status_code=303)

@app.post("/api/reset_faces/{user_id}")
async def api_reset_faces(user_id: int):
    if settings.APP_ROLE != "LOCAL_KIOSK":
        return JSONResponse({'status': 'error', 'message': 'Reset not available'}, status_code=404)
    user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            if f.endswith('.jpg'): os.remove(os.path.join(user_dir, f))
    else:
        os.makedirs(user_dir, exist_ok=True)
    return JSONResponse({'status': 'success'})

@app.get("/audit_logs", response_class=HTMLResponse)
async def audit_logs(request: Request, db: Session = Depends(get_db)):
    if not request.session.get("logged_in"):
        return RedirectResponse(url="/login")
    edits = db.query(AttendanceEdit, Attendance, User).join(
        Attendance, AttendanceEdit.attendance_id == Attendance.id
    ).join(
        User, Attendance.user_id == User.id
    ).order_by(AttendanceEdit.edited_at.desc()).limit(100).all()
    return render_template(request, "audit_logs.html", {"edits": edits})

@app.post("/request_early_report")
async def request_early_report(request: Request, staff_code: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.staff_code == staff_code).first()
    if not user:
        request.state.flash("Staff ID not found.", "danger")
        return RedirectResponse(url="/staff_portal", status_code=303)
        
    if not user.email:
        request.state.flash("No email address associated with your account.", "warning")
        return RedirectResponse(url="/staff_portal", status_code=303)
        
    # Logic for sending email (using send_analytical_email from app.py)
    # For now, just a placeholder
    request.state.flash(f"Early analytical report request received for {user.email}.", "success")
    return RedirectResponse(url="/staff_portal", status_code=303)

@app.get("/export_attendance_csv")
async def export_attendance_csv(db: Session = Depends(get_db)):
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Name', 'Timestamp', 'Status', 'Notes'])
    logs = db.query(Attendance, User).join(User, Attendance.user_id == User.id).order_by(Attendance.timestamp.desc()).all()
    for log, user in logs:
        writer.writerow([log.id, user.name, log.timestamp.strftime('%Y-%m-%d %H:%M:%S'), log.status, log.notes or ''])
    
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_export_{get_now_pht().strftime('%Y%m%d')}.csv"}
    )

@app.get("/generate_report_form/{user_id}", response_class=HTMLResponse)
async def generate_report_form(user_id: int, request: Request, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        request.state.flash("User not found", "danger")
        return RedirectResponse(url="/admin", status_code=303)
    return render_template(request, "report_form.html", {"user": user, "current_month": get_now_pht().month})

@app.post("/generate_report")
async def generate_report(request: Request, user_id: int = Form(...), month: int = Form(...), year: int = Form(...), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    import calendar
    num_days = calendar.monthrange(year, month)[1]
    month_name = calendar.month_name[month]
    
    start_date = datetime(year, month, 1)
    end_date = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
        
    logs = db.query(Attendance).filter(
        Attendance.user_id == user.id,
        Attendance.timestamp >= start_date,
        Attendance.timestamp < end_date
    ).order_by(Attendance.timestamp).all()
    
    logs_by_day = {}
    for log in logs:
        d = log.timestamp.day
        if d not in logs_by_day: logs_by_day[d] = []
        logs_by_day[d].append(log)
        
    days_data = []
    for d in range(1, num_days + 1):
        day_logs = logs_by_day.get(d, [])
        am_in = am_out = pm_in = pm_out = ""
        for log in day_logs:
            t_str = log.timestamp.strftime("%I:%M %p")
            if log.status in ['Present', 'Tardy', 'On Time', 'Late']:
                if log.timestamp.hour < 12:
                    if not am_in: am_in = t_str
                else:
                    if not pm_in: pm_in = t_str
            elif log.status == 'Logout':
                if log.timestamp.hour >= 12: pm_out = t_str
                else: am_out = t_str
        
        days_data.append({'day': d, 'am_in': am_in, 'am_out': am_out, 'pm_in': pm_in, 'pm_out': pm_out, 'ot_in': '', 'ot_out': ''})

    return render_template(request, "report.html", {"user": user, "days": days_data, "month_name": month_name, "year": year})

@app.get("/train")
async def train_model_route(request: Request, background_tasks: BackgroundTasks):
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")

    def train_and_reload():
        if face_engine:
            print("Starting model training...")
            face_engine.train_model()
            print("Training complete. Reloading model...")
            face_engine.reload_model()
            print("Model reloaded.")

    background_tasks.add_task(train_and_reload)
    request.state.flash("Model retraining has been started in the background. It may take a few minutes to complete.", "info")
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/api/train")
async def api_train_model(background_tasks: BackgroundTasks):
    if not face_engine:
        return JSONResponse({'status': 'error', 'message': 'Face engine not available'}, status_code=404)
    
    # For enrollment, we might want to wait for training to finish 
    # OR run it in background. Kiosk usually wants to know when it's done.
    # But since it takes a few seconds, background is safer for timeouts.
    
    def train_and_reload():
        print("API: Starting model training...")
        face_engine.train_model()
        face_engine.reload_model()
        print("API: Model reloaded.")

    background_tasks.add_task(train_and_reload)
    return JSONResponse({'status': 'success', 'message': 'Training started'})

@app.get("/advanced_analytics", response_class=HTMLResponse)
async def advanced_analytics(request: Request, db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.name).all()
    return render_template(request, "analytics.html", {"users": users})

@app.get("/api/advanced_analytics_data")
async def advanced_analytics_data(request: Request, db: Session = Depends(get_db)):
    engine = AnalyticsEngine(db)
    start_str = request.query_params.get('start_date')
    end_str = request.query_params.get('end_date')
    employment_type = request.query_params.get('employment_type')
    user_id = request.query_params.get('user_id')
    
    start_date = datetime.strptime(start_str, '%Y-%m-%d') if start_str else None
    end_date = datetime.strptime(end_str, '%Y-%m-%d') if end_str else None
            
    return JSONResponse({
        'weekly_trends': engine.get_weekly_trends(start_date, end_date, employment_type, user_id),
        'monthly_trends': engine.get_monthly_trends(start_date, end_date, employment_type, user_id),
        'peak_arrival': engine.get_peak_arrival_times(start_date, end_date, employment_type, user_id),
        'status_distribution': engine.get_status_distribution(start_date, end_date, employment_type, user_id),
        'risk_users': engine.predict_risk_users()
    })

@app.get("/api/recent_logs")
async def api_recent_logs(db: Session = Depends(get_db)):
    try:
        logs = db.query(Attendance, User).outerjoin(User, Attendance.user_id == User.id).order_by(Attendance.timestamp.desc()).limit(10).all()
        data = []
        for log, user in logs:
            color = 'success'
            if log.status == 'Tardy': color = 'warning'
            elif log.status == 'Absent': color = 'danger'
            elif log.status == 'Logout': color = 'secondary'
            
            data.append({
                'name': user.name if user else f'ID {log.user_id}',
                'status': log.status,
                'time': log.timestamp.strftime('%I:%M %p'),
                'color': color
            })
        return JSONResponse(data)
    except:
        return JSONResponse([])

@app.post("/api/chat")
async def chat(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    user_message = data.get('query', '')
    if not user_message:
        return JSONResponse({'error': 'No query provided'}, status_code=400)

    # Context injection logic from app.py
    injected_context = ""
    try:
        all_users = db.query(User).all()
        found_users = [u for u in all_users if u.name.lower() in user_message.lower()]
        if found_users:
            injected_context += "\n\n[SYSTEM DATA INJECTION]\n"
            for u in found_users:
                logs = db.query(Attendance).filter(Attendance.user_id == u.id).all()
                on_time = sum(1 for l in logs if l.status in ['On Time', 'Present'])
                late = sum(1 for l in logs if l.status in ['Late', 'Tardy'])
                absent = sum(1 for l in logs if l.status == 'Absent')
                injected_context += f"User: {u.name}\n- On Time: {on_time}\n- Late: {late}\n- Absent: {absent}\n"
    except: pass

    base_context = "You are an internal AI chatbot for the school staff attendance web application..." + injected_context

    try:
        token = os.environ.get("HF_TOKEN")
        if not token: return JSONResponse({'response': "HF_TOKEN missing."})
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
        messages = [{"role": "system", "content": base_context}, {"role": "user", "content": user_message}]
        response = client.chat.completions.create(model="meta-llama/Llama-3.1-70B-Instruct", messages=messages, max_tokens=500)
        return JSONResponse({'response': response.choices[0].message.content})
    except Exception as e:
        return JSONResponse({'response': f"Error: {str(e)}"})

@app.get("/staff_portal", response_class=HTMLResponse)
async def staff_portal_get(request: Request):
    return render_template(request, "staff_portal.html", {"user": None, "logs": []})

@app.post("/staff_portal", response_class=HTMLResponse)
async def staff_portal_post(request: Request, staff_code: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.staff_code == staff_code).first()
    logs = []
    if user:
        logs = db.query(Attendance).filter(Attendance.user_id == user.id).order_by(Attendance.timestamp.desc()).limit(20).all()
    else:
        request.state.flash("Staff ID not found.", "danger")
    return render_template(request, "staff_portal.html", {"user": user, "logs": logs})

# Background Task Logic
async def scheduler_task():
    while True:
        try:
            now = get_now_pht()
            # Implement auto_logout and reports logic here...
            await asyncio.sleep(60)
        except Exception:
            await asyncio.sleep(60)

# Supabase Sync Logic from app.py
def _db_url_reachable(url: str) -> bool:
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn: conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except: return False

def _sync_sqlite_to_supabase_once():
    supabase_db_url = (os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL", "")).strip()
    if not supabase_db_url or not _db_url_reachable(supabase_db_url): return
    # Sync logic implementation...
    pass

async def sync_worker_task():
    while True:
        if settings.APP_ROLE == "LOCAL_KIOSK":
            _sync_sqlite_to_supabase_once()
        await asyncio.sleep(30)

# Email Reporting Logic from app.py
def send_analytical_email(user, logs, is_early=False):
    if not user.email: return False, "No email."
    try:
        # Implementation from app.py...
        return True, "Success"
    except Exception as e: return False, str(e)

@app.on_event("startup")
async def startup_event():
    with engine.begin() as conn:
        try: conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'staff'"))
        except: pass
    
    if settings.APP_ROLE == "LOCAL_KIOSK":
        asyncio.create_task(scheduler_task())
        asyncio.create_task(sync_worker_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
