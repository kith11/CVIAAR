import os
import sys
import cv2
import time
import threading
import base64
import sqlite3
import csv
import io
import random
import pytz
import smtplib
import traceback
import json
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta
from calendar import monthrange
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
# Heavy modules moved into conditional blocks below
# from modules.camera import Camera
# from modules.face_engine import FaceEngine
# from modules.sync_engine import SyncEngine
from modules.analytics_engine import AnalyticsEngine

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("modules.sync_engine").setLevel(logging.INFO)

# --- Core Application Setup ---
# This section initializes the FastAPI application, configures middleware for sessions and
# flash messages, and sets up directories for static files, templates, and data.

app = FastAPI(
    title="CVIAAR - AI-Powered Biometric Attendance",
    description="A real-time attendance monitoring system using facial recognition and liveness detection.",
    version="1.0.0"
)

# --- Middleware Configuration ---
# SessionMiddleware is used for handling user login sessions.
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

class FlashMiddleware(BaseHTTPMiddleware):
    """
    Custom middleware to emulate Flask's flash messaging system.
    It allows routes to store messages in a session and display them on the next request.
    This is useful for showing success or error messages after redirects.
    """
    async def dispatch(self, request: Request, call_next):
        # Function to add a message to the session.
        def flash(message: str, category: str = "info"):
            flashes = request.session.get("_flashes", [])
            flashes.append((category, message))
            request.session["_flashes"] = flashes

        # Function to retrieve and clear messages from the session.
        def get_flashed_messages(with_categories: bool = False):
            flashes = request.session.pop("_flashes", [])
            if with_categories:
                return flashes
            return [f[1] for f in flashes]

        # Make flash functions available in the request state.
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

# --- Database Configuration ---
# The application uses SQLAlchemy for database interaction. It supports both a local SQLite
# database for offline operation (kiosk mode) and a remote PostgreSQL database (e.g., Supabase)
# for a centralized admin dashboard.
APP_ROLE = os.getenv("APP_ROLE", "LOCAL_KIOSK")
DEVICE_ID = os.getenv("DEVICE_ID", "local-device")

# Determine the database URL based on the application's role and environment.
# On Render or similar platforms, DATABASE_URL will point to a remote PostgreSQL.
# In local kiosk mode, we fallback to SQLite.
local_sqlite_url = f"sqlite:///{os.path.join(basedir, 'data', 'offline', 'cviaar_local.sqlite3')}"
db_url = settings.DATABASE_URL or local_sqlite_url

# Fix for SQLAlchemy 1.4+ which requires "postgresql://" instead of "postgres://"
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# The engine is configured with `check_same_thread=False` ONLY for SQLite.
connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
try:
    engine = create_engine(db_url, connect_args=connect_args)
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logging.info(f"Successfully connected to database: {db_url.split('@')[-1] if '@' in db_url else 'SQLite'}")
except Exception as e:
    logging.error(f"Database connection failed: {e}")
    # Fallback to local SQLite if remote fails
    if db_url != local_sqlite_url:
        logging.warning("Falling back to local SQLite database.")
        db_url = local_sqlite_url
        engine = create_engine(db_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables on startup
try:
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing database tables: {e}")

def get_db():
    """FastAPI dependency to create and manage database sessions per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Redis Client for Real-Time Data ---
# Upstash Redis is used as a real-time, low-latency data layer for broadcasting live status
# updates (e.g., who is currently verified) between the kiosk and viewer clients.
redis = None
if settings.UPSTASH_REDIS_URL and settings.UPSTASH_REDIS_TOKEN:
    try:
        redis = Redis(url=settings.UPSTASH_REDIS_URL, token=settings.UPSTASH_REDIS_TOKEN)
        redis.ping() # Test connection
        logging.info("Successfully connected to Upstash Redis.")
    except Exception as e:
        logging.error(f"Error connecting to Upstash Redis: {e}")
        redis = None

def get_redis():
    """FastAPI dependency to provide a Redis client instance."""
    if not redis:
        raise HTTPException(status_code=503, detail="Redis is not configured or available.")
    return redis

# --- Application Modules ---
# These modules are initialized only if the application is running in "LOCAL_KIOSK" mode,
# as they are responsible for hardware interaction (camera) and heavy processing (face engine).
if settings.APP_ROLE == "LOCAL_KIOSK":
    from modules.camera import Camera
    from modules.face_engine import FaceEngine
    from modules.sync_engine import SyncEngine
    
    # Camera module for capturing video frames.
    camera = Camera()
    # FaceEngine handles face detection, recognition, and blink detection.
    face_engine = FaceEngine(
        model_path=os.path.join(basedir, "data", "lbph_model.yml"),
        faces_dir=os.path.join(basedir, "data", "faces"),
    )
    # SyncEngine is a background worker that syncs offline data with a remote server.
    sync_engine = SyncEngine(
        database_url=db_url, # Use the common db_url
        supabase_url=settings.SUPABASE_URL or "",
        supabase_key=settings.SUPABASE_KEY or "",
        remote_db_url=settings.DATABASE_URL, # Use direct Postgres URL if provided
        sync_interval=30, # Sync every 30 seconds
        device_id=settings.DEVICE_ID
    )
    sync_engine.start_sync_worker()
else:
    # In admin mode, these modules are not needed.
    camera = None
    face_engine = None
    sync_engine = None

# --- Global State and Caches ---
# These dictionaries are used for in-memory caching and state management to improve
# performance and handle real-time user interactions.
attendance_cache = {} # Caches the last attendance status for a user to prevent duplicate entries.
recognized_faces = {} # Caches recognized faces to avoid re-processing every frame.
login_attempts = {} # Tracks login attempts to prevent brute-force attacks.
scan_state = {"status": "no_face", "name": None, "timestamp": 0} # Holds the current state of the scanner.
face_verification_cache = {} # Caches liveness verification status based on blinks.

# --- Environment-based Tunables ---
# These functions allow tuning recognition parameters via environment variables without code changes.
def _env_float(key: str, default: float) -> float:
    """Safely reads a float from environment variables, with a fallback default."""
    raw = os.getenv(key)
    if not raw: return default
    try: return float(raw)
    except: return default

def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except:
        return default

# Tunable parameters for face recognition and blink detection.
LBPH_DISTANCE_THRESHOLD = _env_float("LBPH_DISTANCE_THRESHOLD", 50.0)
RECOGNITION_CACHE_TTL_SEC = _env_float("RECOGNITION_CACHE_TTL_SEC", 3.0)
VERIFIED_TTL_SEC = _env_float("VERIFIED_TTL_SEC", 6.0)
MIN_FACE_SIZE_PX = _env_int("MIN_FACE_SIZE_PX", 90)
RECOGNITION_STREAK_REQUIRED = _env_int("RECOGNITION_STREAK_REQUIRED", 2)
RECOGNITION_STREAK_TIMEOUT_SEC = _env_float("RECOGNITION_STREAK_TIMEOUT_SEC", 1.0)

def get_now():
    """Returns the current time in the system's local timezone."""
    return datetime.now().astimezone()

# --- Real-Time Video Streaming ---
def generate_frames():
    """
    An MJPEG frame generator for the live video feed.
    It continuously fetches frames from the camera, encodes them as JPEGs, and yields
    them in a format suitable for streaming to a web browser.
    """
    if not camera:
        # If no camera is found, return a placeholder image with an error message.
        blank = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(blank, "No Camera Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue # Skip if frame is not available
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1 / settings.CAMERA_FPS) # Control frame rate

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Sends an email using the configured Gmail account."""
    if not all([settings.MAIL_USERNAME, settings.MAIL_APP_PASSWORD]):
        logging.error("Email credentials not configured. Skipping email.")
        return False

    msg = MIMEMultipart()
    msg['From'] = settings.MAIL_USERNAME
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(settings.MAIL_USERNAME, settings.MAIL_APP_PASSWORD)
            server.send_message(msg)
            logging.info(f"Email sent successfully to {to_email}")
            return True
    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication Error: Check your MAIL_APP_PASSWORD.")
        return False
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False

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

# --- Core Application Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    """
    Serves the main page.
    - If the app is in "ADMIN_DASHBOARD" mode, it redirects to the admin dashboard.
    - Otherwise, it renders the main monitoring page (index.html).
    """
    if settings.APP_ROLE == "ADMIN_DASHBOARD":
        return RedirectResponse(url="/admin")
    
    users = db.query(User).order_by(User.name).all()
    return render_template(request, "index.html", {"users": users})

@app.get("/video_feed")
async def video_feed():
    """Provides the live MJPEG video stream from the camera."""
    if settings.APP_ROLE != "LOCAL_KIOSK":
        return RedirectResponse(url="/admin")
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Authentication Routes ---

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    """Serves the login page."""
    return render_template(request, "login.html")

@app.post("/login")
async def login_post(request: Request, password: str = Form(...)):
    """
    Handles the login form submission.
    - Validates the admin password.
    - Sets session variables upon successful login.
    - Redirects to the admin dashboard.
    """
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
    """Clears the session to log the user out."""
    request.session.clear()
    request.state.flash("Logged out.", "info")
    return RedirectResponse(url="/", status_code=303)

# --- Admin & Analytics Routes ---

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request, db: Session = Depends(get_db)):
    """
    Serves the main admin dashboard.
    - Requires admin login.
    - Fetches and displays summary data: users, recent logs, attendance trends, and risk predictions.
    """
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")

    analytics = AnalyticsEngine(db)
    users = db.query(User).order_by(User.name).all()
    logs = db.query(Attendance).order_by(Attendance.timestamp.desc()).limit(10).all()
    trends = analytics.get_weekly_trends()
    risks = analytics.predict_risk_users()
    
    return render_template(request, "admin.html", {
        "users": users, 
        "logs": logs, 
        "trends": trends, 
        "risks": risks
    })

@app.get("/advanced_analytics", response_class=HTMLResponse)
async def advanced_analytics(request: Request, db: Session = Depends(get_db)):
    """
    Serves the advanced analytics page.
    - Requires admin login.
    - Provides a more detailed view of attendance data with filtering options.
    """
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")
    
    analytics = AnalyticsEngine(db)
    users = db.query(User).order_by(User.name).all()
    
    return render_template(request, "analytics.html", {
        "users": users,
        "trends": {},
        "stats": {},
        "risk_factors": {},
        "employment_types": ["All", "Full-time", "Part-time", "Contractor"]
    })

@app.get("/edit_user/{user_id}", response_class=HTMLResponse)
async def edit_user_get(request: Request, user_id: int, db: Session = Depends(get_db)):
    """Serves the user profile editing page."""
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")
    
    user = db.get(User, user_id)
    if not user:
        request.state.flash("User not found.", "danger")
        return RedirectResponse(url="/admin", status_code=303)
    
    return render_template(request, "edit_user.html", {"user": user})

@app.post("/update_user/{user_id}")
async def update_user_post(request: Request, user_id: int, 
                           name: str = Form(...),
                           email: str = Form(...),
                           employment_type: str = Form(...),
                           schedule_start: str = Form("06:00"),
                           schedule_end: str = Form("19:00"),
                           role: str = Form("staff"),
                           db: Session = Depends(get_db)):
    """Handles user profile updates."""
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")
    
    user = db.get(User, user_id)
    if not user:
        request.state.flash("User not found.", "danger")
        return RedirectResponse(url="/admin", status_code=303)
    
    # Validate email format
    if not email or '@' not in email or not email.strip():
        request.state.flash("Please enter a valid email address.", "danger")
        return RedirectResponse(url=f"/edit_user/{user_id}", status_code=303)
    
    # Validate Gmail address
    if not email.endswith('@gmail.com'):
        request.state.flash("Please enter a valid Gmail address (must end with @gmail.com).", "danger")
        return RedirectResponse(url=f"/edit_user/{user_id}", status_code=303)
    
    # Check if email is already taken by another user
    existing_user = db.query(User).filter(User.email == email, User.id != user_id).first()
    if existing_user:
        request.state.flash("This email address is already in use by another user.", "danger")
        return RedirectResponse(url=f"/edit_user/{user_id}", status_code=303)
    
    # Update user data
    user.name = name.strip()
    user.email = email.strip()
    user.employment_type = employment_type
    user.schedule_start = schedule_start
    user.schedule_end = schedule_end
    user.role = role
    
    db.commit()
    
    request.state.flash(f"User profile for {user.name} updated successfully.", "success")
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/export_attendance_csv")
async def export_attendance_csv(request: Request, db: Session = Depends(get_db)):
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")

    query = (
        db.query(Attendance, User)
        .join(User, Attendance.user_id == User.id)
        .order_by(Attendance.timestamp.desc())
    )

    def generate():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "attendance_id",
                "user_id",
                "staff_code",
                "name",
                "employment_type",
                "timestamp",
                "status",
            ]
        )
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for attendance, user in query.yield_per(1000):
            ts = attendance.timestamp
            try:
                timestamp_str = ts.astimezone().strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            except Exception:
                try:
                    timestamp_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                except Exception:
                    timestamp_str = str(ts) if ts else ""

            writer.writerow(
                [
                    attendance.id,
                    user.id,
                    user.staff_code or "",
                    user.name or "",
                    user.employment_type or "",
                    timestamp_str,
                    attendance.status or "",
                ]
            )
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    filename = f"attendance_export_{get_now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.get("/generate_report_form")
async def generate_report_form_missing_id(request: Request):
    request.state.flash("Please select a user from the Dashboard to generate a report.", "danger")
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/generate_report_form/{user_id}", response_class=HTMLResponse)
async def generate_report_form(request: Request, user_id: int, db: Session = Depends(get_db)):
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")

    user = db.get(User, user_id)
    if not user:
        request.state.flash("User not found.", "danger")
        return RedirectResponse(url="/admin", status_code=303)

    now = get_now()
    return render_template(
        request,
        "report_form.html",
        {"user": user, "current_month": now.month},
    )

@app.post("/generate_report", response_class=HTMLResponse)
async def generate_report(
    request: Request,
    user_id: int = Form(...),
    month: int = Form(...),
    year: int = Form(...),
    db: Session = Depends(get_db),
):
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    month = max(1, min(12, int(month)))
    year = int(year)

    start = datetime(year, month, 1)
    last_day = monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59, 999999)

    logs = (
        db.query(Attendance)
        .filter(
            Attendance.user_id == user.id,
            Attendance.timestamp >= start,
            Attendance.timestamp <= end,
        )
        .order_by(Attendance.timestamp.asc())
        .all()
    )

    by_day = {}
    for log in logs:
        if not log.timestamp:
            continue
        day = log.timestamp.day
        by_day.setdefault(day, []).append(log)

    in_statuses = {"On Time", "Late", "Present", "Tardy"}
    days = []
    for d in range(1, last_day + 1):
        day_logs = by_day.get(d, [])
        day_logs.sort(key=lambda l: l.timestamp)

        am_in = am_out = pm_in = pm_out = None

        for log in day_logs:
            ts = log.timestamp
            if not ts:
                continue

            is_am = ts.hour < 12
            is_in = (log.status or "") in in_statuses
            is_out = (log.status or "") == "Logout"

            if is_am:
                if is_in and am_in is None:
                    am_in = ts
                elif is_out:
                    am_out = ts
            else:
                if is_in and pm_in is None:
                    pm_in = ts
                elif is_out:
                    pm_out = ts

        fmt = lambda dt: dt.strftime("%I:%M %p") if dt else None
        days.append(
            {
                "day": d,
                "am_in": fmt(am_in),
                "am_out": fmt(am_out),
                "pm_in": fmt(pm_in),
                "pm_out": fmt(pm_out),
                "ot_in": None,
                "ot_out": None,
            }
        )

    month_name = start.strftime("%B")
    return render_template(
        request,
        "report.html",
        {"user": user, "month_name": month_name, "year": year, "days": days},
    )

# --- Staff & Enrollment Routes ---

@app.get("/staff_portal", response_class=HTMLResponse)
async def staff_portal_get(request: Request):
    """Serves the staff portal page where users can view their own attendance."""
    return render_template(request, "staff_portal.html", {"user": None, "logs": []})

@app.post("/staff_portal", response_class=HTMLResponse)
async def staff_portal_post(request: Request, staff_code: str = Form(...), db: Session = Depends(get_db)):
    """
    Handles the staff ID submission and displays the user's attendance records.
    """
    user = db.query(User).filter(User.staff_code == staff_code).first()
    if not user:
        return render_template(request, "staff_portal.html", {"user": None, "logs": [], "error": "Invalid Staff ID"})
    
    logs = db.query(Attendance).filter(Attendance.user_id == user.id).order_by(Attendance.timestamp.desc()).limit(30).all()
    return render_template(request, "staff_portal.html", {"user": user, "logs": logs})

@app.get("/re_enroll", response_class=HTMLResponse)
async def re_enroll_get(request: Request, db: Session = Depends(get_db)):
    """
    Serves a page for admins to select a user to re-enroll.
    """
    if not request.session.get("logged_in") or request.session.get("role") != "admin":
        return RedirectResponse(url="/login")

    users = db.query(User).order_by(User.name).all()
    return render_template(request, "re_enroll.html", {"users": users})

@app.post("/re_enroll")
async def re_enroll_post(request: Request, staff_code: str = Form(...), db: Session = Depends(get_db)):
    """
    Handles the user selection for re-enrollment and redirects to the enrollment page.
    """
    user = db.query(User).filter(User.staff_code == staff_code).first()
    if not user:
        request.state.flash("User not found!", "danger")
        return RedirectResponse(url="/", status_code=303)
    return RedirectResponse(url=f"/enroll/{user.id}?re_enroll=true", status_code=303)

# --- API Endpoints ---

@app.get("/analytics")
async def analytics_endpoint(db: Session = Depends(get_db)):
    """
    Provides a JSON summary of attendance statistics for all users.
    Used by the admin dashboard to populate the main analytics table.
    """
    analytics = AnalyticsEngine(db)
    
    # Get all users
    users = db.query(User).all()
    
    # For each user, calculate their attendance stats
    user_stats = []
    for user in users:
        # Count different attendance statuses for this user
        present_count = db.query(Attendance).filter(
            Attendance.user_id == user.id,
            Attendance.status.in_(["Present", "On Time"])
        ).count()
        
        tardy_count = db.query(Attendance).filter(
            Attendance.user_id == user.id,
            Attendance.status.in_(["Late", "Tardy"])
        ).count()
        
        absent_count = db.query(Attendance).filter(
            Attendance.user_id == user.id,
            Attendance.status == "Absent"
        ).count()
        
        user_stats.append({
            "id": user.id,
            "staff_code": user.staff_code,
            "name": user.name,
            "employment_type": user.employment_type,
            "present": present_count,
            "tardy": tardy_count,
            "absent": absent_count
        })
    
    # Also get weekly trends for the chart
    trends = analytics.get_weekly_trends()
    
    return JSONResponse({
        "stats": user_stats,
        "trends": trends
    })

@app.post("/add_user")
async def add_user(request: Request, name: str = Form(...), email: str = Form(...), employment_type: str = Form("Full-time"), db: Session = Depends(get_db)):
    """
    Handles the creation of a new user from the admin dashboard.
    - Validates input.
    - Generates a unique staff code.
    - Creates the user and redirects to the enrollment page.
    """
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
async def enroll_page(request: Request, user_id: int, db: Session = Depends(get_db), re_enroll: bool = False):
    """
    Serves the face capture page for a specific user.
    - `re_enroll` flag indicates whether to clear existing face data.
    """
    user = db.get(User, user_id)
    if not user:
        request.state.flash("User not found!", "danger")
        return RedirectResponse(url="/admin", status_code=303)
    return render_template(request, "enroll.html", {"user": user, "re_enroll": re_enroll})

@app.post("/api/capture/{user_id}")
async def api_capture(user_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Captures and saves a face image for a user during enrollment.
    - Receives an image (base64-encoded or from the live camera).
    - Detects the most prominent face.
    - Saves the cropped, grayscale face image to the user's directory.
    - Requires `LOCAL_KIOSK` mode.
    """
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
    """
    The core endpoint for face recognition and liveness detection.
    - Receives an image from the frontend.
    - Processes the image using the FaceEngine to detect faces, recognize them, and check for blinks.
    - Implements liveness detection using a time-based verification cache (`face_verification_cache`).
    - Returns a list of detected faces with their recognition status and verification status.
    - Requires `LOCAL_KIOSK` mode.
    """
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

    if recognized_faces:
        stale_keys = []
        for k, v in recognized_faces.items():
            ts = v.get("ts") if isinstance(v, dict) else None
            if ts is None or (now_ts - float(ts)) > RECOGNITION_STREAK_TIMEOUT_SEC:
                stale_keys.append(k)
        for k in stale_keys:
            recognized_faces.pop(k, None)
    
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
        logging.debug(f"Face detection - blink_detected: {res.blink_detected}, bbox: {res.bbox}")

        face_ok = (w >= MIN_FACE_SIZE_PX and h >= MIN_FACE_SIZE_PX)
        candidate = (label != -1 and face_ok and distance < LBPH_DISTANCE_THRESHOLD)

        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        track_key = (int(cx // 80), int(cy // 80))

        stable = False
        if candidate:
            prev = recognized_faces.get(track_key)
            if isinstance(prev, dict) and prev.get("label") == label and (now_ts - float(prev.get("ts", 0))) <= RECOGNITION_STREAK_TIMEOUT_SEC:
                streak = int(prev.get("streak", 0)) + 1
            else:
                streak = 1
            recognized_faces[track_key] = {"label": label, "streak": streak, "ts": now_ts, "distance": float(distance)}
            stable = streak >= RECOGNITION_STREAK_REQUIRED
        else:
            recognized_faces.pop(track_key, None)

        if candidate and not stable:
            result["status"] = "recognized"
            result["name"] = "PLEASE HOLD STILL"

        if candidate and stable:
            user = db.get(User, label)
            if user:
                result['name'] = user.name
                result['status'] = 'recognized'
                result['user_id'] = user.id
                result['confidence'] = distance

                is_currently_verified = face_verification_cache.get(user.id, 0) > (now_ts - VERIFIED_TTL_SEC)

                if res.blink_detected:
                    face_verification_cache[user.id] = now_ts
                    result["verified"] = True
                elif is_currently_verified:
                    result["verified"] = True
                else:
                    result["verified"] = False
        
        results.append(result)

    # Update live status via Redis
    if redis:
        verified_users = {uid: ts for uid, ts in face_verification_cache.items() if ts > (now_ts - VERIFIED_TTL_SEC)}
        try:
            redis.set("live_status:verified_users", json.dumps(verified_users))
        except Exception as e:
            logging.error(f"Redis error on live_status update: {e}")

    return JSONResponse({'status': 'success', 'faces': results})

@app.post("/api/attendance/record")
async def api_attendance_record(request: Request, db: Session = Depends(get_db), redis: Redis = Depends(get_redis)):
    """
    Records an attendance log for a user (login/logout).
    - Requires a verified liveness check.
    - Determines the status (e.g., On Time, Late, Logout) based on the user's schedule.
    - Uses Redis to prevent duplicate login/logout actions.
    - Submits the record to the SyncEngine to be saved to the database.
    """
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

        # --- Cooldown Check ---
        # Prevents rapid, duplicate login/logout actions from the same user.
        last_action_time = attendance_cache.get(uid, 0)
        if now_ts - last_action_time < 60: # 60-second cooldown
            return JSONResponse({'status': 'error', 'message': 'Please wait a moment before your next action.'}, status_code=429)

        now = get_now()
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

        # Update the cooldown timer
        attendance_cache[uid] = now_ts

        return JSONResponse({'status': 'success', 'message': message})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.post("/delete_user/{user_id}")
async def delete_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Deletes a user and all their associated data.
    - Requires admin login.
    - Deletes attendance logs, face data, and the user record.
    - Retrains the face recognition model after deletion.
    """
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

@app.get("/retrain")
@app.post("/retrain")
async def retrain_model(request: Request, background_tasks: BackgroundTasks):
    """
    Initiates a background task to retrain the face recognition model.
    - Requires admin login.
    """
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
    """API endpoint to trigger model retraining as a background task."""
    if face_engine:
        background_tasks.add_task(face_engine.train_model)
        return JSONResponse({'status': 'success', 'message': 'Model training initiated.'})
    else:
        return JSONResponse({'status': 'error', 'message': 'Face engine not available.'}, status_code=500)

@app.get("/api/advanced_analytics_data")
async def advanced_analytics_data(
    start_date: str = None,
    end_date: str = None,
    employment_type: str = "All",
    user_id: str = "All",
    db: Session = Depends(get_db)
):
    """
    Provides detailed analytics data for the advanced analytics page.
    - Fetches data based on the provided filters (date range, employment type, user).
    - Returns weekly trends, monthly trends, status distribution, peak arrival times, and risk predictions.
    """
    analytics = AnalyticsEngine(db)
    
    # Parse dates if provided
    start = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
    end = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None
    
    # Get all the different data slices using the engine
    weekly = analytics.get_weekly_trends(start, end, employment_type, user_id)
    monthly = analytics.get_monthly_trends(start, end, employment_type, user_id)
    
    # Calculate status distribution
    df = analytics.get_attendance_dataframe(start, end, employment_type, user_id)
    status_dist = {"labels": [], "data": []}
    if not df.empty:
        counts = df['status'].value_counts()
        status_dist = {
            "labels": counts.index.tolist(),
            "data": counts.values.tolist()
        }
    
    # Peak arrival calculation
    peak_raw = analytics.get_peak_arrival_times(start, end, employment_type, user_id)
    peak = {"labels": [f"{h}:00" for h in range(24)], "data": [0]*24}
    for h, count in peak_raw.items():
        if 0 <= h < 24:
            peak["data"][h] = int(count)
    
    # Risk users
    risks = analytics.predict_risk_users()
    
    return JSONResponse({
        "weekly_trends": weekly,
        "monthly_trends": monthly,
        "status_distribution": status_dist,
        "peak_arrival": peak,
        "risk_users": risks
    })

@app.post("/request_early_report")
async def request_early_report(request: Request, staff_code: str = Form(...), db: Session = Depends(get_db)):
    """Handles the request for an early attendance report from the staff portal."""
    user = db.query(User).filter(User.staff_code == staff_code).first()
    if not user:
        request.state.flash("Invalid Staff ID", "danger")
        return RedirectResponse(url="/staff_portal", status_code=303)

    if not user.email:
        request.state.flash("No email address on file for this user.", "danger")
        return RedirectResponse(url="/staff_portal", status_code=303)

    # --- Report Analytics ---
    # Calculate summary statistics for the user's recent activity.
    now = get_now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    logs = db.query(Attendance).filter(
        Attendance.user_id == user.id,
        Attendance.timestamp >= start_of_month
    ).order_by(Attendance.timestamp.desc()).all()

    on_time_arrivals = sum(1 for log in logs if log.status in ["On Time", "Present"])
    late_arrivals = sum(1 for log in logs if log.status in ["Late", "Tardy"])
    total_arrivals = on_time_arrivals + late_arrivals
    punctuality_score = (on_time_arrivals / total_arrivals * 100) if total_arrivals > 0 else 100

    # Determine a motivational message based on the score.
    if punctuality_score >= 95:
        motivation = "Outstanding! Your commitment to punctuality is truly commendable. Keep up the fantastic work!"
        motivation_color = "#38A169" # Green
    elif punctuality_score >= 80:
        motivation = "Great job! You have a strong record of being on time. Let's aim for a perfect score!"
        motivation_color = "#3182CE" # Blue
    else:
        motivation = "Every day is a new opportunity. Let's focus on making each arrival a timely one. You can do it!"
        motivation_color = "#DD6B20" # Orange

    # Render the HTML for the email body using the new template.
    template = templates.get_template("email_report.html")
    body = template.render({
        "user": user,
        "logs": logs[:15], # Show the 15 most recent logs in the table
        "report_date": now.strftime('%B %d, %Y'),
        "current_year": now.year,
        "analytics": {
            "on_time": on_time_arrivals,
            "late": late_arrivals,
            "punctuality_score": f"{punctuality_score:.1f}",
            "motivation": motivation,
            "motivation_color": motivation_color
        }
    })

    subject = f"Your Attendance Report for {get_now().strftime('%B %Y')}"

    email_sent = send_email(user.email, subject, body)

    if email_sent:
        request.state.flash(f"Early report request for {user.name} has been sent to {user.email}.", "success")
    else:
        request.state.flash("Could not send email. Please check system configuration.", "danger")

    return RedirectResponse(url="/staff_portal", status_code=303)

@app.post("/api/reset_faces/{user_id}")
async def reset_faces(user_id: int, db: Session = Depends(get_db)):
    """Deletes all face data for a user, allowing for re-enrollment."""
    user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
    if os.path.exists(user_dir):
        import shutil
        shutil.rmtree(user_dir)
        os.makedirs(user_dir, exist_ok=True)
        return JSONResponse({'status': 'success', 'message': f'Faces reset for user {user_id}'})
    return JSONResponse({'status': 'error', 'message': 'User directory not found'}, status_code=404)

@app.get("/api/recent_logs")
async def api_recent_logs(db: Session = Depends(get_db)):
    """
    Provides a JSON list of the 10 most recent attendance logs.
    - Joins with the User table to include the user's name.
    - Used by the main monitoring page to display a live feed of recent activity.
    """
    try:
        logs = db.query(Attendance, User.name).join(User, Attendance.user_id == User.id).order_by(Attendance.timestamp.desc()).limit(10).all()
        
        results = []
        for log, user_name in logs:
            results.append({
                "user_name": user_name,
                "status": log.status,
                "timestamp": log.timestamp.isoformat()
            })
            
        return JSONResponse(results)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/chat")
async def api_chat(request: Request):
    """Handles the AI chatbot interaction, providing a fallback response if offline."""
    data = await request.json()
    user_message = data.get("message", "")
    
    # Simple fallback response if HF_TOKEN is missing or AI fails
    bot_response = "I'm currently in offline mode. How can I help you with the attendance system?"
    
    if settings.HF_TOKEN:
        try:
            # Here we could call Hugging Face API
            # For now, let's keep it simple
            pass
        except Exception:
            pass
            
    return JSONResponse({"response": bot_response})

# --- Main Application Entry Point ---
if __name__ == "__main__":
    """Runs the FastAPI application using uvicorn when the script is executed directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
