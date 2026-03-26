import os
import sys
import logging

# Configure logging early
logging.basicConfig(level=logging.INFO)

# Conditional heavy imports based on APP_ROLE to allow scripts to run without full dependencies
APP_ROLE = os.environ.get("APP_ROLE", "LOCAL_KIOSK")
logging.info(f"APP_ROLE detected: {APP_ROLE}")

if APP_ROLE == "ADMIN_DASHBOARD":
    cv2 = None
else:
    try:
        import cv2
    except ImportError:
        logging.warning("OpenCV (cv2) not found. Camera features will be disabled.")
        cv2 = None

import time
import threading
import base64
import sqlite3
import csv
import io
import random
import math
import shutil
import pytz
import smtplib
import traceback
import json
import asyncio
import uuid
from contextlib import contextmanager
from typing import Optional, List, Any
from datetime import datetime, timedelta, date
from calendar import monthrange
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import FastAPI, Request, Response, Depends, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse, FileResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import create_engine, text, select, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

# Conditional numpy import for lighter script runs
if APP_ROLE != "ADMIN_DASHBOARD":
    try:
        import numpy as np
    except ImportError:
        np = None
else:
    np = None

# Conditional redis imports for lighter script runs
try:
    from upstash_redis import Redis as UpstashRedis
except ImportError:
    UpstashRedis = None

try:
    import redis as local_redis
except ImportError:
    local_redis = None

from config import settings
from modules.models import Base, User, Attendance, AttendanceEdit, ExcuseNote, ensure_attendance_schema
# Heavy modules moved into conditional blocks below
# from modules.camera import Camera
# from modules.face_engine import FaceEngine
# from modules.sync_engine import SyncEngine
from modules.analytics_engine import AnalyticsEngine
from modules.attendance_rules import (
    ABSENT_STATUSES,
    infer_event_type,
    is_present_status,
    normalize_session,
)
from modules.runtime_state import ThreadSafeTTLStore

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("modules.sync_engine").setLevel(logging.INFO)

# --- Thread-Safe Runtime State --- #
analytics_cache = ThreadSafeTTLStore(maxsize=100, default_ttl=300)
risk_cache = ThreadSafeTTLStore(maxsize=16, default_ttl=3600)

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
INSECURE_SECRET_KEYS = {
    "",
    "a-very-secret-key-that-you-should-change",
    "supersecretkey_change_this_in_production",
}
INSECURE_ADMIN_PASSWORDS = {"", "admin123"}
SESSION_ROLE_KEY = "role"
STAFF_SESSION_USER_ID_KEY = "staff_user_id"
ADMIN_SESSION_FLAG_KEY = "logged_in"


def normalize_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().strip('"').strip("'")
    return normalized or None


def normalize_database_url(url: str | None) -> str | None:
    normalized = normalize_env_value(url)
    if normalized and normalized.startswith("postgres://"):
        normalized = normalized.replace("postgres://", "postgresql://", 1)
    return normalized


def describe_database_connection_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "tenant or user not found" in message or "password authentication failed" in message:
        return "invalid database credentials or wrong Supabase pooler username"
    if "could not translate host name" in message or "name or service not known" in message:
        return "database host name could not be resolved"
    if "ssl" in message:
        return "SSL negotiation with the database failed"
    if "timed out" in message or "timeout" in message:
        return "database connection timed out"
    if "connection refused" in message or "could not connect to server" in message:
        return "database server is unreachable"
    return "unexpected database connection failure"


def get_database_url_hint(database_url: str | None) -> str | None:
    normalized = normalize_database_url(database_url)
    if not normalized:
        return None

    if ".pooler.supabase.com:6543" in normalized or ".pooler.supabase.com:5432" in normalized:
        return (
            "Supabase pooler URLs must be copied exactly from the Supabase Connect panel. "
            "For pooler hosts, the username is usually project-specific, such as "
            "'postgres.[PROJECT_REF]'. If Render is a long-running server, prefer the direct "
            "database URL when your network supports it, or the Supavisor session pooler URL."
        )

    if ".supabase.co:5432" in normalized and "@db." in normalized:
        return (
            "This looks like a direct Supabase database URL. Ensure the host, password, and SSL "
            "settings match the current values in the Supabase Connect panel."
        )

    return None


def validate_runtime_settings(config=settings) -> None:
    if config.APP_ROLE != "ADMIN_DASHBOARD":
        return

    issues = []
    if not normalize_database_url(config.DATABASE_URL):
        issues.append("DATABASE_URL is required in ADMIN_DASHBOARD mode.")
    if normalize_env_value(config.SECRET_KEY) in INSECURE_SECRET_KEYS:
        issues.append("SECRET_KEY must be replaced with a strong production secret.")
    if normalize_env_value(config.ADMIN_PASSWORD) in INSECURE_ADMIN_PASSWORDS:
        issues.append("ADMIN_PASSWORD must be replaced with a strong non-default value.")

    if issues:
        raise RuntimeError(" ".join(issues))


validate_runtime_settings()

session_cookie_secure = settings.APP_ROLE == "ADMIN_DASHBOARD" and not settings.DEBUG
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    same_site="lax",
    https_only=session_cookie_secure,
    max_age=settings.SESSION_MAX_AGE_SECONDS,
    session_cookie="cviaar_session",
)

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

# --- Instance Lifecycle Management ---
# To prevent multiple instances of the application from running concurrently,
# we use a PID file. If a PID file already exists, we check if the process is still running.
PID_FILE = os.path.join(basedir, "data", "app.pid")

def manage_instance_lifecycle():
    # Only manage PID if we are in the main worker process, not the reloader or a CLI script
    if os.environ.get("UVICORN_INTERACTIVE") == "true" or \
       os.environ.get("UVICORN_RELOADER_PROCESS") == "true" or \
       os.environ.get("APP_ROLE") == "ADMIN_DASHBOARD":
        return

    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Empty PID file")
                old_pid = int(content)
            
            # Check if the process is still running
            try:
                os.kill(old_pid, 0)
                logging.error(f"Another instance of the application is already running (PID: {old_pid}). Exiting.")
                sys.exit(1)
            except OSError:
                # Process is not running, we can safely overwrite the PID file
                logging.info(f"Found orphaned PID file for PID {old_pid}. Overwriting.")
        except (ValueError, IOError, PermissionError) as e:
            logging.warning(f"Could not read PID file: {e}. Overwriting.")

    try:
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
        
        # Ensure PID file is removed on exit
        import atexit
        atexit.register(lambda: os.path.exists(PID_FILE) and os.remove(PID_FILE))
        logging.info(f"PID file created at {PID_FILE} (PID: {os.getpid()})")
    except Exception as e:
        logging.error(f"Failed to create PID file: {e}")

manage_instance_lifecycle()

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
DEVICE_ID = os.getenv("DEVICE_ID", "local-device")

# In kiosk mode we always keep the primary write path on local SQLite.
# In admin dashboard mode we require a working direct database connection.
sqlite_path = normalize_env_value(settings.SQLITE_DB_PATH) or os.path.join("data", "offline", "cviaar_local.sqlite3")
if not os.path.isabs(sqlite_path):
    sqlite_path = os.path.join(basedir, sqlite_path)
local_sqlite_url = f"sqlite:///{sqlite_path}"
remote_database_url = normalize_database_url(settings.DATABASE_URL)
supabase_rest_url = normalize_env_value(settings.SUPABASE_URL)
supabase_rest_key = normalize_env_value(settings.SUPABASE_KEY)


def create_checked_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    checked_engine = create_engine(database_url, connect_args=connect_args)
    with checked_engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return checked_engine


db_url = remote_database_url if settings.APP_ROLE == "ADMIN_DASHBOARD" else local_sqlite_url
try:
    engine = create_checked_engine(db_url)
    target_label = "remote database" if db_url != local_sqlite_url else "local SQLite database"
    logging.info("Successfully connected to %s.", target_label)
except Exception as e:
    diagnosis = describe_database_connection_error(e)
    logging.error("Database connection failed (%s): %s", diagnosis, e)
    hint = get_database_url_hint(db_url)
    if hint:
        logging.error("Database URL hint: %s", hint)
    if settings.APP_ROLE == "ADMIN_DASHBOARD":
        raise RuntimeError(
            f"ADMIN_DASHBOARD startup aborted: {diagnosis}. "
            "Set a valid DATABASE_URL on Render or Supabase."
        ) from e
    logging.warning("Continuing in LOCAL_KIOSK mode with offline SQLite only.")
    db_url = local_sqlite_url
    engine = create_checked_engine(db_url)

if settings.APP_ROLE == "LOCAL_KIOSK":
    logging.info(
        "Kiosk sync configuration: direct_db=%s, supabase_rest_fallback=%s",
        "enabled" if remote_database_url else "disabled",
        "enabled" if supabase_rest_url and supabase_rest_key else "disabled",
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables on startup
try:
    Base.metadata.create_all(bind=engine)
    ensure_attendance_schema(engine)
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
# If Upstash is not configured, we fall back to a standard local Redis.
redis = None
if settings.UPSTASH_REDIS_URL and settings.UPSTASH_REDIS_TOKEN and UpstashRedis:
    try:
        redis = UpstashRedis(url=settings.UPSTASH_REDIS_URL, token=settings.UPSTASH_REDIS_TOKEN)
        redis.ping() # Test connection
        logging.info("Successfully connected to Upstash Redis.")
    except Exception as e:
        logging.error(f"Error connecting to Upstash Redis: {e}")
        redis = None

if not redis and settings.REDIS_URL and local_redis:
    try:
        redis = local_redis.from_url(settings.REDIS_URL, decode_responses=True)
        redis.ping()
        logging.info(f"Successfully connected to local Redis at {settings.REDIS_URL}")
    except Exception as e:
        logging.error(f"Error connecting to local Redis: {e}")
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
    if settings.SYNC_ENABLED:
        sync_engine = SyncEngine(
            database_url=local_sqlite_url,
            supabase_url=supabase_rest_url or "",
            supabase_key=supabase_rest_key or "",
            remote_db_url=remote_database_url,
            sync_interval=30, # Sync every 30 seconds
            device_id=settings.DEVICE_ID
        )
        sync_engine.start_sync_worker()
    else:
        sync_engine = None
        logging.info("Sync engine is disabled in configuration.")
else:
    # In admin mode, these modules are not needed.
    camera = None
    face_engine = None
    sync_engine = None

# --- Environment-based Tunables ---
# These functions allow tuning recognition parameters via environment variables without code changes.
def _env_float(key: str, default: float) -> float:
    """Safely reads a float from environment variables, with a fallback default."""
    raw = os.getenv(key)
    if not raw: return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default

def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

# Tunable parameters for face recognition and blink detection.
LBPH_DISTANCE_THRESHOLD = _env_float("LBPH_DISTANCE_THRESHOLD", 50.0)
RECOGNITION_CACHE_TTL_SEC = _env_float("RECOGNITION_CACHE_TTL_SEC", 3.0)
VERIFIED_TTL_SEC = _env_float("VERIFIED_TTL_SEC", 6.0)
MIN_FACE_SIZE_PX = _env_int("MIN_FACE_SIZE_PX", 90)
RECOGNITION_STREAK_REQUIRED = _env_int("RECOGNITION_STREAK_REQUIRED", 2)
RECOGNITION_STREAK_TIMEOUT_SEC = _env_float("RECOGNITION_STREAK_TIMEOUT_SEC", 1.0)
ATTENDANCE_DEDUP_WINDOW_SEC = _env_float("ATTENDANCE_DEDUP_WINDOW_SEC", 5.0)
ENROLLMENT_CAPTURE_TARGET = _env_int("ENROLLMENT_CAPTURE_TARGET", 40)
ENROLLMENT_POSE_BUCKETS = ("center", "left", "right", "up", "down")
ENROLLMENT_BUCKET_TARGET = max(4, ENROLLMENT_CAPTURE_TARGET // len(ENROLLMENT_POSE_BUCKETS))

# --- Global State and Caches ---
attendance_cache = ThreadSafeTTLStore(maxsize=1000, default_ttl=max(ATTENDANCE_DEDUP_WINDOW_SEC * 2, 10.0))
recognized_faces = ThreadSafeTTLStore(maxsize=512, default_ttl=RECOGNITION_STREAK_TIMEOUT_SEC)
login_attempts = ThreadSafeTTLStore(maxsize=200, default_ttl=300.0)
scan_state = ThreadSafeTTLStore(maxsize=1)
scan_state.set("latest", {"status": "no_face", "name": None, "timestamp": 0})
face_verification_cache = ThreadSafeTTLStore(maxsize=512, default_ttl=VERIFIED_TTL_SEC)
recognition_rate_limit = ThreadSafeTTLStore(maxsize=1000, default_ttl=2.0)

model_operation_lock = threading.Lock()
model_mutation_lock = threading.Lock()


class ModelBusyError(RuntimeError):
    """Raised when a vision model operation cannot proceed safely."""


@contextmanager
def model_operation(timeout: float | None = None, message: str = "Face model is busy. Please try again."):
    if not face_engine:
        yield
        return

    acquired = model_operation_lock.acquire() if timeout is None else model_operation_lock.acquire(timeout=timeout)
    if not acquired:
        raise ModelBusyError(message)

    try:
        yield
    finally:
        model_operation_lock.release()


def try_begin_model_mutation() -> bool:
    if not face_engine:
        return True
    return model_mutation_lock.acquire(blocking=False)


def end_model_mutation() -> None:
    if face_engine and model_mutation_lock.locked():
        model_mutation_lock.release()


def safe_commit(db: Session, context: str) -> bool:
    try:
        db.commit()
        return True
    except SQLAlchemyError as exc:
        db.rollback()
        logging.error("%s failed: %s", context, exc)
        return False


def generate_staff_code(db: Session, max_attempts: int = 25) -> str:
    for _ in range(max_attempts):
        candidate = f"{random.randint(0, 999999):06d}"
        exists = db.query(User.id).filter(User.staff_code == candidate).first()
        if not exists:
            return candidate
    raise RuntimeError("Unable to generate a unique staff code.")


def get_enrollment_pose_counts(user_dir: str) -> dict[str, int]:
    counts = {bucket: 0 for bucket in ENROLLMENT_POSE_BUCKETS}
    if not os.path.exists(user_dir):
        return counts

    for name in os.listdir(user_dir):
        if not name.lower().endswith(".jpg"):
            continue
        prefix = name.split("_", 1)[0].lower()
        bucket = prefix if prefix in counts else "center"
        counts[bucket] += 1
    return counts


def get_enrollment_total_count(user_dir: str) -> int:
    if not os.path.exists(user_dir):
        return 0
    return sum(1 for name in os.listdir(user_dir) if name.lower().endswith(".jpg"))


def suggest_next_pose(pose_counts: dict[str, int]) -> str:
    priorities = {bucket: index for index, bucket in enumerate(ENROLLMENT_POSE_BUCKETS)}
    return min(
        ENROLLMENT_POSE_BUCKETS,
        key=lambda bucket: (pose_counts.get(bucket, 0), priorities[bucket]),
    )


def pose_instruction(bucket: str) -> str:
    instructions = {
        "center": "Look straight at the camera.",
        "left": "Turn your face slightly to the left.",
        "right": "Turn your face slightly to the right.",
        "up": "Lift your chin slightly upward.",
        "down": "Lower your chin slightly downward.",
    }
    return instructions.get(bucket, "Look straight at the camera.")


def is_capture_too_similar(user_dir: str, pose_bucket: str, face_img: Any) -> bool:
    candidates = []
    for name in os.listdir(user_dir):
        if not name.lower().endswith(".jpg"):
            continue
        prefix = name.split("_", 1)[0].lower()
        if prefix == pose_bucket:
            path = os.path.join(user_dir, name)
            candidates.append((os.path.getmtime(path), path))

    if not candidates:
        return False

    latest_path = max(candidates, key=lambda item: item[0])[1]
    previous = cv2.imread(latest_path, cv2.IMREAD_GRAYSCALE)
    if previous is None:
        return False

    current = face_engine.preprocess_face(face_img)
    previous = face_engine.preprocess_face(previous)
    delta = float(np.mean(cv2.absdiff(previous, current)))
    return delta < 8.0


def get_present_set_key(target_date: date | None = None) -> str:
    target = target_date or get_now().date()
    return f"attendance:{target.isoformat()}:present"


def build_attendance_fields(
    status: str,
    timestamp: datetime | None = None,
    session: str | None = None,
    event_type: str | None = None,
    auto_generated: bool = False,
) -> dict[str, Any]:
    record_ts = timestamp or get_now()
    return {
        "timestamp": record_ts,
        "session": normalize_session(session, record_ts),
        "event_type": infer_event_type(status, event_type),
        "auto_generated": 1 if auto_generated else 0,
        "status": status,
    }


def latest_user_status_for_day(db: Session, user_id: int, target_date: date) -> str | None:
    start = datetime.combine(target_date, datetime.min.time())
    end = datetime.combine(target_date, datetime.max.time())
    latest = (
        db.query(Attendance)
        .filter(
            Attendance.user_id == user_id,
            Attendance.timestamp >= start,
            Attendance.timestamp <= end,
        )
        .order_by(Attendance.timestamp.desc())
        .first()
    )
    return latest.status if latest else None


def count_present_users_from_db(db: Session, target_date: date | None = None) -> int:
    date_value = target_date or get_now().date()
    start = datetime.combine(date_value, datetime.min.time())
    end = datetime.combine(date_value, datetime.max.time())
    logs = (
        db.query(Attendance)
        .filter(Attendance.timestamp >= start, Attendance.timestamp <= end)
        .order_by(Attendance.user_id.asc(), Attendance.timestamp.desc())
        .all()
    )

    latest_status_by_user: dict[int, str] = {}
    for log in logs:
        if log.user_id not in latest_status_by_user:
            latest_status_by_user[log.user_id] = log.status

    return sum(1 for status in latest_status_by_user.values() if is_present_status(status))


def get_live_present_count(db: Session) -> int:
    key = get_present_set_key()
    if redis:
        try:
            return int(redis.scard(key))
        except Exception as exc:
            logging.warning("Falling back to DB live-status count after Redis error: %s", exc)
    return count_present_users_from_db(db)


def record_attendance_direct(
    db: Session,
    user_id: int,
    status: str,
    timestamp: datetime | None = None,
    session: str | None = None,
    event_type: str | None = None,
    auto_generated: bool = False,
) -> bool:
    fields = build_attendance_fields(
        status,
        timestamp=timestamp,
        session=session,
        event_type=event_type,
        auto_generated=auto_generated,
    )
    record = Attendance(
        sync_key=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=fields["timestamp"],
        status=fields["status"],
        session=fields["session"],
        event_type=fields["event_type"],
        auto_generated=fields["auto_generated"],
        device_id=settings.DEVICE_ID,
        synced=0,
    )
    db.add(record)
    return safe_commit(db, f"Attendance record for user {user_id}")


REPORT_EMAIL_COOLDOWN_MINUTES = 10


def format_relative_time(dt_value: datetime | None) -> str | None:
    if not dt_value:
        return None

    now = get_now()
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=now.tzinfo)

    delta = now - dt_value
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = seconds // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def get_report_email_state(user: User) -> dict[str, Any]:
    now = get_now()
    email_ready = bool(user.email and all([settings.MAIL_USERNAME, settings.MAIL_APP_PASSWORD]))
    cooldown_remaining = 0

    if user.last_report_sent:
        last_sent = user.last_report_sent
        if last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=now.tzinfo)
        elapsed_seconds = (now - last_sent).total_seconds()
        cooldown_remaining = max(0, REPORT_EMAIL_COOLDOWN_MINUTES * 60 - int(elapsed_seconds))

    return {
        "email_ready": email_ready,
        "mail_configured": bool(all([settings.MAIL_USERNAME, settings.MAIL_APP_PASSWORD])),
        "recipient": user.email,
        "last_sent_at": user.last_report_sent,
        "last_sent_label": format_relative_time(user.last_report_sent),
        "cooldown_remaining": cooldown_remaining,
        "can_send_now": email_ready and cooldown_remaining == 0,
    }


def build_staff_portal_context(db: Session, user: User | None = None, error: str | None = None) -> dict[str, Any]:
    context: dict[str, Any] = {"user": user, "logs": [], "error": error}
    if not user:
        return context

    logs = (
        db.query(Attendance)
        .filter(Attendance.user_id == user.id)
        .order_by(Attendance.timestamp.desc())
        .limit(30)
        .all()
    )
    context["logs"] = logs

    login_logs = [log for log in logs if infer_event_type(log.status, getattr(log, "event_type", None)) == "login"]
    on_time_count = sum(1 for log in login_logs if log.status in {"Present", "On Time"})
    late_count = sum(1 for log in login_logs if log.status in {"Late", "Tardy"})
    absent_count = sum(1 for log in logs if log.status in ABSENT_STATUSES)
    context["summary"] = {
        "records_count": len(logs),
        "on_time_count": on_time_count,
        "late_count": late_count,
        "absent_count": absent_count,
        "latest_log": logs[0] if logs else None,
    }
    context["report_email"] = get_report_email_state(user)
    context["report_month"] = get_now().month
    context["report_year"] = get_now().year
    return context


def build_report_days_for_user(user: User, month: int, year: int, db: Session) -> tuple[datetime, list[dict[str, Any]]]:
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

            session_name = normalize_session(getattr(log, "session", None), ts)
            is_in = (log.status or "") in in_statuses
            is_out = (log.status or "") == "Logout"
            is_session_absent = (log.status or "") in {"AM Absent", "PM Absent"}
            is_legacy_day_absent = (log.status or "") == "Absent" and getattr(log, "session", None) is None

            if session_name == "AM":
                if is_in and am_in is None:
                    am_in = ts
                elif is_out:
                    am_out = ts
                elif is_session_absent and am_in is None:
                    am_in = "Absent"
            elif session_name == "PM":
                if is_in and pm_in is None:
                    pm_in = ts
                elif is_out:
                    pm_out = ts
                elif is_session_absent and pm_in is None:
                    pm_in = "Absent"

            if is_legacy_day_absent:
                if am_in is None:
                    am_in = "Absent"
                if pm_in is None:
                    pm_in = "Absent"

        def fmt(value: Any) -> str | None:
            if isinstance(value, datetime):
                return value.strftime("%I:%M %p")
            if isinstance(value, str):
                return value
            return None

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

    return start, days


def queue_model_retrain(background_tasks: BackgroundTasks) -> bool:
    if not face_engine:
        return False
    if not try_begin_model_mutation():
        return False
    background_tasks.add_task(train_and_reload_model_task)
    return True


def train_and_reload_model_task() -> None:
    try:
        if not face_engine:
            return
        with model_operation():
            trained = face_engine.train_model()
            if trained:
                face_engine.reload_model()
            else:
                logging.warning("Model retraining completed without training samples.")
    except Exception as exc:
        logging.error("Background model retraining failed: %s", exc)
    finally:
        end_model_mutation()


def queue_email_delivery(background_tasks: BackgroundTasks, to_email: str, subject: str, body: str) -> None:
    background_tasks.add_task(send_email_task, to_email, subject, body)


def send_email_task(to_email: str, subject: str, body: str) -> None:
    if not send_email(to_email, subject, body):
        logging.error("Background email delivery failed for %s", to_email)

def get_now():
    """Returns the current time in the system's local timezone."""
    return datetime.now().astimezone()

# --- Real-Time Global State ---
camera_active = False
current_session_user = None

# --- Real-Time Video Streaming ---
def generate_frames():
    """
    An MJPEG frame generator for the live video feed.
    It continuously fetches frames from the camera, encodes them as JPEGs, and yields
    them in a format suitable for streaming to a web browser.
    """
    global camera_active
    
    while True:
        if not camera or not camera_active:
            # If camera is inactive, return a dark placeholder frame
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "CAMERA STANDBY", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
            continue

        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1 / settings.CAMERA_FPS)

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
def render_template(request: Request, name: str, context: dict | None = None):
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
    ctx.update(context or {})
    return templates.TemplateResponse(name, ctx)


def clear_admin_session(request: Request) -> None:
    request.session.pop(ADMIN_SESSION_FLAG_KEY, None)
    if request.session.get(SESSION_ROLE_KEY) == "admin":
        request.session.pop(SESSION_ROLE_KEY, None)


def clear_staff_session(request: Request) -> None:
    request.session.pop(STAFF_SESSION_USER_ID_KEY, None)
    if request.session.get(SESSION_ROLE_KEY) == "staff":
        request.session.pop(SESSION_ROLE_KEY, None)


def set_admin_session(request: Request) -> None:
    request.session.clear()
    request.session[ADMIN_SESSION_FLAG_KEY] = True
    request.session[SESSION_ROLE_KEY] = "admin"


def set_staff_session(request: Request, user_id: int) -> None:
    request.session.clear()
    request.session[SESSION_ROLE_KEY] = "staff"
    request.session[STAFF_SESSION_USER_ID_KEY] = int(user_id)


def is_admin_authenticated(request: Request) -> bool:
    return bool(
        request.session.get(ADMIN_SESSION_FLAG_KEY)
        and request.session.get(SESSION_ROLE_KEY) == "admin"
    )


def get_staff_session_user_id(request: Request) -> int | None:
    if request.session.get(SESSION_ROLE_KEY) != "staff":
        return None
    raw = request.session.get(STAFF_SESSION_USER_ID_KEY)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def get_staff_session_user(request: Request, db: Session) -> User | None:
    user_id = get_staff_session_user_id(request)
    if user_id is None:
        return None
    user = db.get(User, user_id)
    if not user:
        clear_staff_session(request)
        return None
    return user


def require_admin_page(request: Request) -> Response | None:
    if is_admin_authenticated(request):
        return None
    return RedirectResponse(url="/login", status_code=303)


def require_admin_api(request: Request) -> Response | None:
    if is_admin_authenticated(request):
        return None
    return JSONResponse({"status": "error", "message": "Admin authentication required."}, status_code=403)


def require_staff_page(request: Request, db: Session) -> tuple[User | None, Response | None]:
    user = get_staff_session_user(request, db)
    if user:
        return user, None
    return None, RedirectResponse(url="/staff_portal", status_code=303)

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

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Provides a default favicon to avoid 404 errors in the browser console."""
    favicon_path = os.path.join(basedir, "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    # Return a 204 No Content if no favicon is found to suppress the error
    return Response(status_code=204)

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
    if is_admin_authenticated(request):
        return RedirectResponse(url="/admin", status_code=303)
    return render_template(request, "login.html")

@app.post("/login")
async def login_post(request: Request, password: str = Form(...)):
    """
    Handles the login form submission.
    - Validates the admin password.
    - Sets session variables upon successful login.
    - Redirects to the admin dashboard.
    """
    admin_password = settings.ADMIN_PASSWORD
    if password == admin_password:
        set_admin_session(request)
        request.state.flash("Logged in successfully.", "success")
        return RedirectResponse(url="/admin", status_code=303)
    else:
        request.state.flash("Invalid password.", "danger")
        return render_template(request, "login.html")

@app.get("/staff_logout")
async def staff_logout(request: Request):
    """Clears only the staff session and returns to the staff portal login screen."""
    clear_staff_session(request)
    request.state.flash("Signed out of the staff portal.", "info")
    return RedirectResponse(url="/staff_portal", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    """Clears the session to log the user out."""
    current_role = request.session.get(SESSION_ROLE_KEY)
    request.session.clear()
    request.state.flash("Logged out.", "info")
    redirect_target = "/staff_portal" if current_role == "staff" else "/"
    return RedirectResponse(url=redirect_target, status_code=303)

# --- Admin & Analytics Routes ---

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request, db: Session = Depends(get_db)):
    """
    Serves the main admin dashboard.
    - Requires admin login.
    - Fetches and displays summary data: users, recent logs, attendance trends, and risk predictions.
    """
    guard = require_admin_page(request)
    if guard:
        return guard

    try:
        analytics = AnalyticsEngine(db)
        users = db.query(User).order_by(User.name).all()
        
        # Only fetch logs that have an associated user to avoid template errors (log.user.name)
        # Using inner join to filter out orphaned logs
        logs = db.query(Attendance).join(User, Attendance.user_id == User.id).order_by(Attendance.timestamp.desc()).limit(10).all()
        
        # Safeguard analytics results for template rendering
        try:
            trends = analytics.get_weekly_trends()
        except Exception as e:
            logging.warning(f"Failed to fetch weekly trends: {e}")
            trends = {"labels": [], "present": [], "tardy": [], "absent": []}

        try:
            risks = analytics.predict_risk_users()
        except Exception as e:
            logging.warning(f"Failed to predict risk users: {e}")
            risks = []
        
        return render_template(request, "admin.html", {
            "users": users, 
            "logs": logs, 
            "trends": trends, 
            "risks": risks
        })
    except Exception as e:
        logging.error(f"Error in admin dashboard: {e}")
        logging.error(traceback.format_exc())
        return HTMLResponse(content="Internal Server Error: See logs for details.", status_code=500)

@app.get("/audit_logs", response_class=HTMLResponse)
async def audit_logs(request: Request, db: Session = Depends(get_db)):
    """
    Displays the audit logs for attendance modifications.
    - Requires admin login.
    """
    guard = require_admin_page(request)
    if guard:
        return guard
    
    try:
        # Fetch edits joined with attendance logs and users
        edits = db.query(AttendanceEdit, Attendance, User).join(
            Attendance, AttendanceEdit.attendance_id == Attendance.id
        ).join(
            User, Attendance.user_id == User.id
        ).order_by(AttendanceEdit.edited_at.desc()).all()
        
        return render_template(request, "audit_logs.html", {
            "edits": edits
        })
    except Exception as e:
        logging.error(f"Error in audit logs: {e}")
        logging.error(traceback.format_exc())
        return HTMLResponse(content="Internal Server Error: See logs for details.", status_code=500)

@app.get("/api/users")
async def api_get_users(request: Request, db: Session = Depends(get_db)):
    """API endpoint to fetch all users for selection lists."""
    guard = require_admin_api(request)
    if guard:
        return guard
    users = db.query(User).all()
    return [{"id": u.id, "name": u.name} for u in users]

@app.get("/advanced_analytics", response_class=HTMLResponse)
async def advanced_analytics(request: Request, db: Session = Depends(get_db)):
    """
    Serves the advanced analytics page.
    - Requires admin login.
    - Provides a more detailed view of attendance data with filtering options.
    """
    guard = require_admin_page(request)
    if guard:
        return guard
    
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
    guard = require_admin_page(request)
    if guard:
        return guard
    
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
    guard = require_admin_page(request)
    if guard:
        return guard
    
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

    if not safe_commit(db, f"Update user {user_id}"):
        request.state.flash("Could not update the user profile right now.", "danger")
        return RedirectResponse(url=f"/edit_user/{user_id}", status_code=303)

    request.state.flash(f"User profile for {user.name} updated successfully.", "success")
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/export_attendance_csv")
async def export_attendance_csv(request: Request, db: Session = Depends(get_db)):
    guard = require_admin_page(request)
    if guard:
        return guard

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
                "session",
                "event_type",
                "auto_generated",
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
                    attendance.session or "",
                    attendance.event_type or "",
                    int(attendance.auto_generated or 0),
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
    guard = require_admin_page(request)
    if guard:
        return guard

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
    guard = require_admin_page(request)
    if guard:
        return guard

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    month = max(1, min(12, int(month)))
    year = int(year)
    start, days = build_report_days_for_user(user, month, year, db)

    month_name = start.strftime("%B")
    return render_template(
        request,
        "report.html",
        {"user": user, "month_name": month_name, "year": year, "days": days},
    )

# --- Staff & Enrollment Routes ---

@app.get("/staff_portal", response_class=HTMLResponse)
async def staff_portal_get(request: Request, db: Session = Depends(get_db)):
    """Serves the staff portal page where users can view their own attendance."""
    if is_admin_authenticated(request):
        return RedirectResponse(url="/admin", status_code=303)
    user = get_staff_session_user(request, db)
    return render_template(request, "staff_portal.html", build_staff_portal_context(db, user))

@app.post("/staff_portal", response_class=HTMLResponse)
async def staff_portal_post(request: Request, 
                            staff_code: str = Form(""),
                            db: Session = Depends(get_db)):
    """
    Authenticates a staff member using their 6-digit staff code.
    """
    if is_admin_authenticated(request):
        return RedirectResponse(url="/admin", status_code=303)

    code = (staff_code or "").strip()
    if not code:
        return render_template(request, "staff_portal.html", build_staff_portal_context(db, None, "Enter your 6-digit staff code."))

    user = db.query(User).filter(User.staff_code == code).first()
    if not user:
        clear_staff_session(request)
        return render_template(request, "staff_portal.html", build_staff_portal_context(db, None, "Invalid staff code. Please try again."))

    set_staff_session(request, user.id)
    request.state.flash(f"Welcome back, {user.name}.", "success")
    return RedirectResponse(url="/staff_portal", status_code=303)


@app.get("/staff_report/{user_id}", response_class=HTMLResponse)
async def staff_report(user_id: int, request: Request, month: int | None = None, year: int | None = None, db: Session = Depends(get_db)):
    """Allows a staff user to view a printable monthly report from the staff portal."""
    user, guard = require_staff_page(request, db)
    if guard:
        return guard
    if not user or user.id != user_id:
        request.state.flash("You can only open your own staff report.", "danger")
        return RedirectResponse(url="/staff_portal", status_code=303)

    now = get_now()
    report_month = max(1, min(12, int(month or now.month)))
    report_year = int(year or now.year)
    start, days = build_report_days_for_user(user, report_month, report_year, db)
    return render_template(
        request,
        "report.html",
        {"user": user, "month_name": start.strftime("%B"), "year": report_year, "days": days},
    )

@app.get("/re_enroll", response_class=HTMLResponse)
async def re_enroll_get(request: Request, db: Session = Depends(get_db)):
    """
    Serves a page for admins to select a user to re-enroll.
    """
    guard = require_admin_page(request)
    if guard:
        return guard

    users = db.query(User).order_by(User.name).all()
    return render_template(request, "re_enroll.html", {"users": users})

@app.post("/re_enroll")
async def re_enroll_post(request: Request, staff_code: str = Form(...), db: Session = Depends(get_db)):
    """
    Handles the user selection for re-enrollment and redirects to the enrollment page.
    """
    guard = require_admin_page(request)
    if guard:
        return guard
    user = db.query(User).filter(User.staff_code == staff_code).first()
    if not user:
        request.state.flash("User not found!", "danger")
        return RedirectResponse(url="/", status_code=303)
    return RedirectResponse(url=f"/enroll/{user.id}?re_enroll=true", status_code=303)

# --- API Endpoints ---

@app.get("/analytics")
async def analytics_endpoint(request: Request, db: Session = Depends(get_db)):
    """
    Provides a JSON summary of attendance statistics for all users.
    Used by the admin dashboard to populate the main analytics table.
    """
    try:
        analytics = AnalyticsEngine(db)
        
        # Get all users
        users = db.query(User).all()
        
        # For each user, calculate their attendance stats (last 7 days for consistency with trends)
        user_stats = []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        start_dt = datetime.combine(start_date, datetime.min.time())

        for user in users:
            present_count = db.query(Attendance).filter(
                Attendance.user_id == user.id,
                Attendance.status.in_(["Present", "On Time"]),
                Attendance.timestamp >= start_dt
            ).count()
            
            tardy_count = db.query(Attendance).filter(
                Attendance.user_id == user.id,
                Attendance.status.in_(["Late", "Tardy"]),
                Attendance.timestamp >= start_dt
            ).count()
            
            absent_count = db.query(Attendance).filter(
                Attendance.user_id == user.id,
                Attendance.status.in_(list(ABSENT_STATUSES)),
                Attendance.timestamp >= start_dt
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
        trends = analytics.get_weekly_trends(start_date, end_date)
        
        return JSONResponse({
            "status": "success",
            "stats": user_stats,
            "trends": trends
        })
    except Exception as e:
        logging.error(f"Dashboard analytics error: {e}")
        logging.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e), "traceback": traceback.format_exc()}, status_code=500)

@app.post("/add_user")
async def add_user(request: Request, name: str = Form(...), email: str = Form(...), employment_type: str = Form("Full-time"), db: Session = Depends(get_db)):
    """
    Handles the creation of a new user from the admin dashboard.
    - Validates input.
    - Generates a unique staff code.
    - Creates the user and redirects to the enrollment page.
    """
    guard = require_admin_page(request)
    if guard:
        return guard

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

    new_user = None
    for _ in range(5):
        try:
            staff_code = generate_staff_code(db)
            new_user = User(
                name=name.strip(),
                email=email.strip(),
                employment_type=employment_type,
                staff_code=staff_code,
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            break
        except IntegrityError:
            db.rollback()
            logging.warning("Staff code collision detected while creating %s. Retrying.", email)
            new_user = None
        except SQLAlchemyError as exc:
            db.rollback()
            logging.error("Failed to create user %s: %s", email, exc)
            request.state.flash("Could not create the user right now.", "danger")
            return RedirectResponse(url="/admin", status_code=303)

    if not new_user:
        request.state.flash("Could not generate a unique staff code. Please try again.", "danger")
        return RedirectResponse(url="/admin", status_code=303)

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
    return render_template(request, "enroll.html", {
        "user": user,
        "re_enroll": re_enroll,
        "capture_target": ENROLLMENT_CAPTURE_TARGET,
        "pose_bucket_target": ENROLLMENT_BUCKET_TARGET,
    })

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
    guard = require_admin_api(request)
    if guard:
        return guard
    
    user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
    if not os.path.exists(user_dir):
        return JSONResponse({'status': 'error', 'message': 'User directory not found'}, status_code=404)

    existing = get_enrollment_total_count(user_dir)
    pose_counts = get_enrollment_pose_counts(user_dir)
    if existing >= ENROLLMENT_CAPTURE_TARGET:
        return JSONResponse({
            'status': 'complete',
            'count': existing,
            'target': ENROLLMENT_CAPTURE_TARGET,
            'guidance': "Enrollment set complete.",
        })

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

    guard = require_admin_api(request)
    if guard:
        return guard

    try:
        with model_operation(timeout=0.25, message="Face model is being updated. Please wait a moment."):
            face_results = face_engine.detect_faces(frame)
    except ModelBusyError as exc:
        return JSONResponse({'status': 'busy', 'message': str(exc)}, status_code=503)

    if face_results:
        # Sort by bbox size to get the most prominent face
        face_results = sorted(face_results, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        res = face_results[0]
        (x, y, w, h) = res.bbox
        
        # Ensure coordinates are within frame boundaries
        y1, y2 = max(0, y), min(frame.shape[0], y+h)
        x1, x2 = max(0, x), min(frame.shape[1], x+w)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return JSONResponse({
                'status': 'retry',
                'count': existing,
                'target': ENROLLMENT_CAPTURE_TARGET,
                'message': "Face crop was empty. Please re-center your face.",
                'guidance': pose_instruction(suggest_next_pose(pose_counts)),
            })

        quality = face_engine.assess_capture_quality(face_img, res.landmarks_2d)
        pose_bucket = quality.get("pose_bucket", "center")

        if not quality.get("ok"):
            return JSONResponse({
                'status': 'retry',
                'count': existing,
                'target': ENROLLMENT_CAPTURE_TARGET,
                'message': quality.get("reason", "Capture quality too low."),
                'guidance': pose_instruction(suggest_next_pose(pose_counts)),
                'pose_bucket': pose_bucket,
            })

        if pose_counts.get(pose_bucket, 0) >= ENROLLMENT_BUCKET_TARGET:
            return JSONResponse({
                'status': 'retry',
                'count': existing,
                'target': ENROLLMENT_CAPTURE_TARGET,
                'message': f"We have enough {pose_bucket} images. {pose_instruction(suggest_next_pose(pose_counts))}",
                'guidance': pose_instruction(suggest_next_pose(pose_counts)),
                'pose_bucket': pose_bucket,
            })

        if is_capture_too_similar(user_dir, pose_bucket, face_img):
            return JSONResponse({
                'status': 'retry',
                'count': existing,
                'target': ENROLLMENT_CAPTURE_TARGET,
                'message': "That frame is too similar to the previous one. Change your head angle slightly.",
                'guidance': pose_instruction(suggest_next_pose(pose_counts)),
                'pose_bucket': pose_bucket,
            })

        processed_face = face_engine.preprocess_face(face_img)
        filename = f"{pose_bucket}_{existing + 1:03d}.jpg"
        cv2.imwrite(os.path.join(user_dir, filename), processed_face)

        updated_count = existing + 1
        updated_pose_counts = get_enrollment_pose_counts(user_dir)
        next_pose = suggest_next_pose(updated_pose_counts)
        return JSONResponse({
            'status': 'success',
            'count': updated_count,
            'target': ENROLLMENT_CAPTURE_TARGET,
            'message': f"Captured {pose_bucket} pose successfully.",
            'guidance': pose_instruction(next_pose),
            'pose_bucket': pose_bucket,
            'pose_counts': updated_pose_counts,
        })
    
    return JSONResponse({
        'status': 'no_face',
        'count': existing,
        'target': ENROLLMENT_CAPTURE_TARGET,
        'message': 'No face detected. Please look directly at the camera.',
        'guidance': pose_instruction(suggest_next_pose(pose_counts)),
    })

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
    
    # Rate limiting check
    client_ip = request.client.host
    current_count = recognition_rate_limit.increment(client_ip, delta=1, ttl=2.0, initial=0)
    if current_count > 20:
        return JSONResponse({'status': 'error', 'message': 'Too many requests'}, status_code=429)

    # If camera is not active, don't process recognition
    if not camera_active:
        return JSONResponse({'status': 'idle', 'faces': []})

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

    now_ts = time.time()
    results = []

    try:
        with model_operation(timeout=0.25, message="Face model is being updated. Please wait a moment."):
            face_results = face_engine.process_frame(frame)

            for res in face_results:
                x, y, w, h = res.bbox
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

                logging.debug(f"Face detection - blink_detected: {res.blink_detected}, bbox: {res.bbox}")

                face_ok = (w >= MIN_FACE_SIZE_PX and h >= MIN_FACE_SIZE_PX)
                candidate = (label != -1 and face_ok and distance < LBPH_DISTANCE_THRESHOLD)

                cx = x + (w / 2.0)
                cy = y + (h / 2.0)
                track_key = (int(cx // 80), int(cy // 80))

                stable = False
                if candidate:
                    prev = recognized_faces.get(track_key)
                    if isinstance(prev, dict) and prev.get("label") == label:
                        streak = int(prev.get("streak", 0)) + 1
                    else:
                        streak = 1
                    recognized_faces.set(
                        track_key,
                        {"label": label, "streak": streak, "ts": now_ts, "distance": float(distance)},
                        ttl=RECOGNITION_STREAK_TIMEOUT_SEC,
                    )
                    stable = streak >= RECOGNITION_STREAK_REQUIRED
                else:
                    recognized_faces.remove(track_key)

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
                            face_verification_cache.set(user.id, now_ts, ttl=VERIFIED_TTL_SEC)
                            result["verified"] = True
                        elif is_currently_verified:
                            result["verified"] = True
                        else:
                            result["verified"] = False

                results.append(result)
    except ModelBusyError as exc:
        return JSONResponse({'status': 'busy', 'message': str(exc), 'faces': []}, status_code=503)

    # Update live status via Redis
    if redis:
        verified_users = face_verification_cache.snapshot()
        try:
            redis.set("live_status:verified_users", json.dumps(verified_users))
        except Exception as e:
            logging.error(f"Redis error on live_status update: {e}")

    return JSONResponse({'status': 'success', 'faces': results})

@app.post("/api/camera/state")
async def set_camera_state(request: Request):
    """Activates or deactivates the camera from the frontend."""
    global camera_active, current_session_user
    data = await request.json()
    state = data.get("active", False)
    camera_active = state
    if not state:
        current_session_user = None
    return JSONResponse({"status": "success", "active": camera_active})

@app.post("/api/attendance/record")
async def api_attendance_record(request: Request, db: Session = Depends(get_db)):
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

        # --- Short same-action dedupe check ---
        # Prevents duplicate submits from double clicks or repeated frontend requests
        # without blocking legitimate follow-up actions like immediate logout after login.
        action_normalized = action.lower()
        last_action_entry = attendance_cache.get(uid)
        if isinstance(last_action_entry, dict):
            previous_action = str(last_action_entry.get("action", "")).lower()
            previous_ts = float(last_action_entry.get("ts", 0))
            if previous_action == action_normalized and (now_ts - previous_ts) < ATTENDANCE_DEDUP_WINDOW_SEC:
                return JSONResponse({'status': 'error', 'message': 'This action was just recorded. Please wait a few seconds and try again.'}, status_code=429)

        now = get_now()
        today_str = now.strftime("%Y-%m-%d")
        live_set_key = f"attendance:{today_str}:present"
        attendance_fields = build_attendance_fields(
            "Logout" if action_normalized == "logout" else "Pending",
            timestamp=now,
            event_type=action_normalized,
        )
        redis_client = redis
        redis_available = redis_client is not None

        if action_normalized == 'login':
            already_logged_in = False
            if redis_available:
                try:
                    already_logged_in = bool(redis_client.sismember(live_set_key, uid))
                except Exception as exc:
                    logging.warning("Redis login presence check failed for user %s: %s", uid, exc)
                    redis_available = False
            if not redis_available:
                already_logged_in = is_present_status(latest_user_status_for_day(db, uid, now.date()))

            if already_logged_in:
                return JSONResponse({'status': 'error', 'message': 'Already logged in.'}, status_code=409)

            sch_start_str = user.schedule_start or "06:00"
            try:
                sch_start_dt = now.replace(hour=int(sch_start_str[:2]), minute=int(sch_start_str[3:]), second=0, microsecond=0)
            except (TypeError, ValueError, IndexError):
                sch_start_dt = now.replace(hour=6, minute=0)

            status = 'On Time' if now <= sch_start_dt + timedelta(minutes=15) else 'Late'
            attendance_fields["status"] = status

            if sync_engine:
                try:
                    sync_engine.record_attendance(
                        user_id=uid,
                        status=status,
                        timestamp=attendance_fields["timestamp"],
                        session=attendance_fields["session"],
                        event_type=attendance_fields["event_type"],
                    )
                except Exception as exc:
                    logging.error("SyncEngine record failed for user %s: %s", uid, exc)
                    return JSONResponse({'status': 'error', 'message': 'Could not record attendance right now.'}, status_code=500)
            elif not record_attendance_direct(
                db,
                uid,
                status,
                timestamp=attendance_fields["timestamp"],
                session=attendance_fields["session"],
                event_type=attendance_fields["event_type"],
            ):
                return JSONResponse({'status': 'error', 'message': 'Could not record attendance right now.'}, status_code=500)

            if redis_available:
                try:
                    redis_client.sadd(live_set_key, uid)
                except Exception as exc:
                    logging.warning("Redis login presence update failed for user %s: %s", uid, exc)

            message = f"Login successful for {user.name}."

        elif action_normalized == 'logout':
            if sync_engine:
                try:
                    sync_engine.record_attendance(
                        user_id=uid,
                        status='Logout',
                        timestamp=attendance_fields["timestamp"],
                        session=attendance_fields["session"],
                        event_type=attendance_fields["event_type"],
                    )
                except Exception as exc:
                    logging.error("SyncEngine logout record failed for user %s: %s", uid, exc)
                    return JSONResponse({'status': 'error', 'message': 'Could not record attendance right now.'}, status_code=500)
            elif not record_attendance_direct(
                db,
                uid,
                'Logout',
                timestamp=attendance_fields["timestamp"],
                session=attendance_fields["session"],
                event_type=attendance_fields["event_type"],
            ):
                return JSONResponse({'status': 'error', 'message': 'Could not record attendance right now.'}, status_code=500)

            if redis_available:
                try:
                    redis_client.srem(live_set_key, uid)
                except Exception as exc:
                    logging.warning("Redis logout presence update failed for user %s: %s", uid, exc)

            message = f"Logout successful for {user.name}."
        
        else:
            return JSONResponse({'status': 'error', 'message': 'Invalid action'}, status_code=400)

        # Update the cooldown timer
        attendance_cache.set(uid, {"action": action_normalized, "ts": now_ts}, ttl=max(ATTENDANCE_DEDUP_WINDOW_SEC * 2, 10.0))

        return JSONResponse({'status': 'success', 'message': message})

    except Exception as e:
        logging.error("Attendance record error: %s", e)
        logging.error(traceback.format_exc())
        return JSONResponse({'status': 'error', 'message': 'Unexpected attendance error.'}, status_code=500)

@app.get("/api/live-status")
async def api_live_status(db: Session = Depends(get_db)):
    """Returns the current number of present users for the admin dashboard."""
    try:
        return JSONResponse({
            "status": "success",
            "present_count": get_live_present_count(db),
        })
    except Exception as exc:
        logging.error("Live status error: %s", exc)
        return JSONResponse({"status": "error", "message": "Could not load live status."}, status_code=500)

@app.post("/delete_user/{user_id}")
async def delete_user(user_id: int, request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Deletes a user and all their associated data.
    - Requires admin login.
    - Deletes attendance logs, face data, and the user record.
    - Retrains the face recognition model after deletion.
    """
    guard = require_admin_page(request)
    if guard:
        return guard

    mutation_started = try_begin_model_mutation()
    if face_engine and not mutation_started:
        request.state.flash("Face model is busy updating. Please try deleting the user again in a moment.", "warning")
        return RedirectResponse(url="/admin", status_code=303)

    guard = require_admin_page(request)
    if guard:
        return guard

    user = db.get(User, user_id)
    if user:
        db.query(Attendance).filter(Attendance.user_id == user.id).delete()
        db.delete(user)
        if not safe_commit(db, f"Delete user {user_id}"):
            if face_engine:
                end_model_mutation()
            request.state.flash("Could not delete the user right now.", "danger")
            return RedirectResponse(url="/admin", status_code=303)

        user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
        try:
            with model_operation(timeout=0.25, message="Face model is being updated. Please try again shortly."):
                if os.path.exists(user_dir):
                    shutil.rmtree(user_dir)
        except ModelBusyError:
            if face_engine:
                end_model_mutation()
            request.state.flash("Face model is busy updating. Please try deleting the user again in a moment.", "warning")
            return RedirectResponse(url="/admin", status_code=303)
        except OSError as exc:
            if face_engine:
                end_model_mutation()
            logging.error("Failed to remove face directory for user %s: %s", user_id, exc)
            request.state.flash("User data was removed, but face files could not be cleaned up.", "warning")
            return RedirectResponse(url="/admin", status_code=303)

        if face_engine:
            background_tasks.add_task(train_and_reload_model_task)

        request.state.flash(f"User {user.name} has been deleted.", "success")
    else:
        if face_engine:
            end_model_mutation()
        request.state.flash("User not found.", "danger")
        
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/retrain")
@app.post("/retrain")
async def retrain_model(request: Request, background_tasks: BackgroundTasks):
    """
    Initiates a background task to retrain the face recognition model.
    - Requires admin login.
    """
    guard = require_admin_page(request)
    if guard:
        return guard
        
    if face_engine:
        if queue_model_retrain(background_tasks):
            request.state.flash("Model retraining has been initiated. This may take a few minutes.", "info")
        else:
            request.state.flash("Face model is already updating. Please wait for the current run to finish.", "warning")
    else:
        request.state.flash("Face engine not available.", "danger")
        
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/api/train")
async def api_train_model(request: Request, background_tasks: BackgroundTasks):
    """API endpoint to trigger model retraining as a background task."""
    guard = require_admin_api(request)
    if guard:
        return guard
    if face_engine:
        if queue_model_retrain(background_tasks):
            return JSONResponse({'status': 'success', 'message': 'Model training initiated.'})
        return JSONResponse({'status': 'busy', 'message': 'Face model is already updating.'}, status_code=409)
    else:
        return JSONResponse({'status': 'error', 'message': 'Face engine not available.'}, status_code=500)

@app.get("/api/advanced_analytics_data")
async def advanced_analytics_data(
    request: Request,
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
    guard = require_admin_api(request)
    if guard:
        return guard
    # Create a cache key based on the query parameters
    cache_key = f"{start_date}-{end_date}-{employment_type}-{user_id}"

    cached_result = analytics_cache.get(cache_key)
    if cached_result is not None:
        logging.info(f"Serving analytics data from cache for key: {cache_key}")
        return JSONResponse(cached_result)

    logging.info(f"Cache miss. Generating new analytics data for key: {cache_key}")
    analytics = AnalyticsEngine(db)
    
    # Parse dates if provided
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date and start_date != "null" else None
        end = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date and end_date != "null" else None
    except ValueError:
        start, end = None, None
    
    # Get all the different data slices using the engine
    weekly = analytics.get_weekly_trends(start, end, employment_type, user_id)
    monthly = analytics.get_monthly_trends(start, end, employment_type, user_id)
    
    # Calculate status distribution
    status_dist = analytics.get_status_distribution(start, end, employment_type, user_id)
    
    # Peak arrival calculation
    peak_raw = analytics.get_peak_arrival_times(start, end, employment_type, user_id)
    peak = {"labels": [f"{h:02d}:00" for h in range(24)], "data": [0]*24}
    for h, count in peak_raw.items():
        if 0 <= h < 24:
            peak["data"][h] = int(count)
    
    # Risks users - with separate caching
    cached_risks = risk_cache.get("risks")
    if cached_risks is not None:
        risks = cached_risks
    else:
        risks = analytics.predict_risk_users()
        risk_cache.set("risks", risks)
    
    # Store the result in the cache
    result = {
        "weekly_trends": weekly,
        "monthly_trends": monthly,
        "status_distribution": status_dist,
        "peak_arrival": peak,
        "risk_users": risks,
        "insights": [] # Initialize empty list for insights
    }

    # Generate insights only if they are not already cached or if requested
    insights = analytics.get_advanced_insights(start, end, employment_type, user_id)
    result["insights"] = insights
    
    analytics_cache.set(cache_key, result)
    
    return JSONResponse(result)

@app.post("/request_early_report")
async def request_early_report(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: int = Form(...),
    month: int = Form(...),
    year: int = Form(...),
    db: Session = Depends(get_db),
):
    """Handles the request for an early attendance report from the staff portal."""
    staff_user, guard = require_staff_page(request, db)
    if guard:
        return guard
    if not staff_user:
        return RedirectResponse(url="/staff_portal", status_code=303)
    if user_id != staff_user.id:
        request.state.flash("You can only request your own attendance report.", "danger")
        return RedirectResponse(url="/staff_portal", status_code=303)
    user = staff_user

    if not user.email:
        request.state.flash("No email address on file for this user.", "danger")
        return RedirectResponse(url="/staff_portal", status_code=303)

    report_email = get_report_email_state(user)
    if not report_email["mail_configured"]:
        request.state.flash("Email sending is not configured on this device yet.", "danger")
        return RedirectResponse(url="/staff_portal", status_code=303)

    if report_email["cooldown_remaining"] > 0:
        wait_minutes = max(1, math.ceil(report_email["cooldown_remaining"] / 60))
        request.state.flash(
            f"A report was just requested for {user.name}. Please wait about {wait_minutes} minute{'s' if wait_minutes != 1 else ''} before sending another one.",
            "warning",
        )
        return RedirectResponse(url="/staff_portal", status_code=303)

    # --- Report Analytics ---
    # Calculate summary statistics for the user's recent activity.
    now = get_now()
    report_month = max(1, min(12, int(month)))
    report_year = int(year)
    start_of_month = now.replace(year=report_year, month=report_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    month_last_day = monthrange(report_year, report_month)[1]
    end_of_month = now.replace(year=report_year, month=report_month, day=month_last_day, hour=23, minute=59, second=59, microsecond=999999)
    
    logs = db.query(Attendance).filter(
        Attendance.user_id == user.id,
        Attendance.timestamp >= start_of_month,
        Attendance.timestamp <= end_of_month
    ).order_by(Attendance.timestamp.desc()).all()

    analytics_engine = AnalyticsEngine(db)
    
    # Comprehensive analytics for the user
    end_date = end_of_month.date()
    start_date = start_of_month.date()
    
    # Arrival patterns
    peak_times = analytics_engine.get_peak_arrival_times(start_date, end_date, user_id=user.id)
    peak_hour = max(peak_times, key=peak_times.get) if peak_times else None
    peak_hour_fmt = f"{peak_hour:02d}:00" if peak_hour is not None else "N/A"
    
    # Weekly comparison
    weekly_comp = analytics_engine._get_weekly_comparison(end_date - timedelta(days=7), end_date, None, user.id)
    
    on_time_arrivals = sum(1 for log in logs if log.status in ["On Time", "Present"])
    late_arrivals = sum(1 for log in logs if log.status in ["Late", "Tardy"])
    absent_sessions = sum(1 for log in logs if log.status in ABSENT_STATUSES)
    total_arrivals = on_time_arrivals + late_arrivals
    punctuality_score = (on_time_arrivals / total_arrivals * 100) if total_arrivals > 0 else 100

    # Determine a motivational message based on the score.
    if punctuality_score >= 95:
        motivation = "Outstanding! Your commitment to punctuality is truly commendable. Keep up the fantastic work!"
        motivation_color = "#198754" # Success Green
    elif punctuality_score >= 80:
        motivation = "Great job! You have a strong record of being on time. Let's aim for a perfect score!"
        motivation_color = "#0d6efd" # Primary Blue
    else:
        motivation = "Every day is a new opportunity. Let's focus on making each arrival a timely one. You can do it!"
        motivation_color = "#fd7e14" # Warning Orange

    # Render the HTML for the email body using the new template.
    template = templates.get_template("email_report.html")
    body = template.render({
        "user": user,
        "logs": logs[:15], # Show the 15 most recent logs in the table
        "report_date": start_of_month.strftime('%B %Y'),
        "current_year": now.year,
        "analytics": {
            "on_time": on_time_arrivals,
            "late": late_arrivals,
            "absent": absent_sessions,
            "punctuality_score": f"{punctuality_score:.1f}",
            "motivation": motivation,
            "motivation_color": motivation_color,
            "peak_hour": peak_hour_fmt,
            "weekly_growth": weekly_comp.get("growth", 0),
            "engagement_change": weekly_comp.get("engagement_change", 0)
        }
    })

    subject = f"Your Attendance Report for {start_of_month.strftime('%B %Y')}"

    queue_email_delivery(background_tasks, user.email, subject, body)
    user.last_report_sent = now
    if not safe_commit(db, f"Update last report sent for user {user.id}"):
        request.state.flash("The report was queued, but the send status could not be saved.", "warning")
        return RedirectResponse(url="/staff_portal", status_code=303)

    request.state.flash(
        f"Attendance report for {user.name} was queued for delivery to {user.email}. Please allow a few minutes for arrival.",
        "success",
    )

    return RedirectResponse(url="/staff_portal", status_code=303)

@app.post("/api/reset_faces/{user_id}")
async def reset_faces(user_id: int, request: Request, db: Session = Depends(get_db)):
    """Deletes all face data for a user, allowing for re-enrollment."""
    guard = require_admin_api(request)
    if guard:
        return guard
    if face_engine and not try_begin_model_mutation():
        return JSONResponse({'status': 'busy', 'message': 'Face model is already updating.'}, status_code=409)

    user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
    try:
        with model_operation(timeout=0.25, message="Face model is being updated. Please try again shortly."):
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
                os.makedirs(user_dir, exist_ok=True)
                return JSONResponse({'status': 'success', 'message': f'Faces reset for user {user_id}'})
            return JSONResponse({'status': 'error', 'message': 'User directory not found'}, status_code=404)
    except ModelBusyError as exc:
        return JSONResponse({'status': 'busy', 'message': str(exc)}, status_code=503)
    finally:
        if face_engine:
            end_model_mutation()

@app.get("/api/recent_logs")
async def api_recent_logs(request: Request, db: Session = Depends(get_db)):
    """
    Provides a JSON list of the 10 most recent attendance logs.
    - Joins with the User table to include the user's name.
    - Used by the main monitoring page to display a live feed of recent activity.
    """
    guard = require_admin_api(request)
    if guard:
        return guard

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
        logging.error("Recent logs error: %s", e)
        logging.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": "Could not load recent logs."}, status_code=500)

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
