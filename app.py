import os
import sys
import cv2
import time
import threading
import base64
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash, session
from functools import wraps
from sqlalchemy.exc import SQLAlchemyError
from modules.models import db, User, Attendance, AttendanceEdit, ExcuseNote
from modules.camera import Camera
from modules.face_engine import FaceEngine
from modules.analytics_engine import AnalyticsEngine
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

load_dotenv()

app = Flask(__name__)

if getattr(sys, "frozen", False):
    basedir = os.path.dirname(sys.executable)
else:
    basedir = os.path.abspath(os.path.dirname(__file__))

APP_ROLE = os.getenv("APP_ROLE", "LOCAL_KIOSK")
DEVICE_ID = os.getenv("DEVICE_ID", "local-device")

supabase_db_url = os.getenv("SUPABASE_DB_URL")
if not supabase_db_url:
    raise RuntimeError("SUPABASE_DB_URL environment variable is required but not set.")
app.config["SQLALCHEMY_DATABASE_URI"] = supabase_db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = os.getenv("SECRET_KEY", "default_dev_key")
app.config["ADMIN_PASSWORD"] = os.getenv("ADMIN_PASSWORD", "admin123")

# Security Settings
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,  # Set to False if not using HTTPS
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

@app.before_request
def make_session_permanent():
    session.permanent = True

# Global Error Handler
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error for the developer
    app.logger.error(f"Unhandled Exception: {str(e)}", exc_info=True)
    # Return a generic message to the user
    return jsonify({
        "status": "error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500

os.makedirs(os.path.join(basedir, "data", "faces"), exist_ok=True)
os.makedirs(os.path.join(basedir, "data", "offline"), exist_ok=True)

db.init_app(app)

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Kiosk Access decorator
def kiosk_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Admin is always allowed
        if session.get('logged_in'):
            return f(*args, **kwargs)
        # Check kiosk auth
        if not session.get('kiosk_authorized'):
            return redirect(url_for('kiosk_login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize Camera and Face Engine
if APP_ROLE == "LOCAL_KIOSK":
    camera = Camera()
    face_engine = FaceEngine(
        model_path=os.path.join(basedir, "data", "lbph_model.yml"),
        faces_dir=os.path.join(basedir, "data", "faces"),
    )
else:
    camera = None
    face_engine = None

# Global variables
attendance_cache = {}
recognized_faces = {}
login_attempts = {} # Rate limiting for login
scan_state = {"status": "no_face", "name": None, "timestamp": 0}
verified_live_users = {}
face_verification_cache = {}
first_blink_times = {}
AUTO_LOGOUT_TIME = "19:30"

def get_current_date():
    return datetime.now().strftime('%Y-%m-%d')


OFFLINE_DB_PATH = os.path.join(basedir, "data", "offline", "attendance_queue.db")


def init_offline_db():
    if APP_ROLE != "LOCAL_KIOSK":
        return
    conn = sqlite3.connect(OFFLINE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            device_id TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def enqueue_offline_attendance(user_id, timestamp, status):
    if APP_ROLE != "LOCAL_KIOSK":
        return
    conn = sqlite3.connect(OFFLINE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO pending_attendance (user_id, timestamp, status, device_id) VALUES (?, ?, ?, ?)",
        (user_id, timestamp.isoformat(), status, DEVICE_ID),
    )
    conn.commit()
    conn.close()


def sync_offline_attendance():
    if APP_ROLE != "LOCAL_KIOSK":
        return
    if not os.path.exists(OFFLINE_DB_PATH):
        return
    conn = sqlite3.connect(OFFLINE_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, user_id, timestamp, status, device_id FROM pending_attendance ORDER BY id"
    )
    rows = cur.fetchall()
    for row_id, user_id, ts_str, status, device_id in rows:
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            ts = datetime.now()
        rec = Attendance(
            user_id=user_id,
            status=status,
            timestamp=ts,
            device_id=device_id,
        )
        db.session.add(rec)
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            conn.close()
            return
        cur.execute("DELETE FROM pending_attendance WHERE id = ?", (row_id,))
        conn.commit()
    conn.close()

def mark_attendance(user_id):
    today = get_current_date()
    
    if user_id in attendance_cache and attendance_cache[user_id] == today:
        return False
    
    with app.app_context():
        existing = Attendance.query.filter_by(user_id=user_id).filter(
            db.func.date(Attendance.timestamp) == datetime.now().date()
        ).first()
        
        if existing:
            attendance_cache[user_id] = today
            return False

        user = db.session.get(User, user_id)
        if not user:
            return False

        now = datetime.now()
        
        sch_start_str = user.schedule_start or "06:00"
        try:
            sch_start_dt = datetime.strptime(f"{today} {sch_start_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            sch_start_dt = datetime.strptime(f"{today} 06:00", "%Y-%m-%d %H:%M")
        
        on_time_end = sch_start_dt + timedelta(hours=1, minutes=15)
        
        window_start = sch_start_dt - timedelta(hours=1)
        
        if window_start <= now <= on_time_end:
            status = 'On Time'
        elif now > on_time_end:
            status = 'Late'
        else:
            status = 'On Time'
            
        new_record = Attendance(
            user_id=user_id,
            status=status,
            timestamp=now,
            device_id=DEVICE_ID,
        )
        db.session.add(new_record)
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            enqueue_offline_attendance(user_id, now, status)
        attendance_cache[user_id] = today
        print(f"Marked {status} for user {user_id}")
        return True


@app.before_request
def sync_offline_before_request():
    if APP_ROLE != "LOCAL_KIOSK":
        return
    try:
        sync_offline_attendance()
    except SQLAlchemyError:
        db.session.rollback()
    except Exception:
        pass

def generate_frames():
    frame_count = 0
    faces_data = []
    
    # Consistency Check Variables
    current_face_id = None
    consecutive_face_count = 0
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            # Yield a blank frame or error image if camera is not working
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "No Camera Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)
            continue

        frame_count += 1
        
        # Optimize: Only process faces every 3rd frame
        if frame_count % 3 == 0:
            # Face Detection with Mesh (for Liveness)
            faces_data = face_engine.detect_faces_mesh(frame)
        
        # Scan State logic variables
        found_verified = False
        found_recognized = False
        found_unknown = False
        detected_name = None
        
        # Reuse faces_data for frames in between
        for ((x, y, w, h), landmarks) in faces_data:
            # Recognize
            face_img = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            label, confidence = face_engine.recognize_face(gray_face)
            
            text = "Unknown"
            status_color = (0, 0, 255) # Red by default
            status_text = ""
            
            # Consistency Logic
            is_consistent = False
            
            # DEBUG: Print confidence to console
            if label != -1:
                print(f"DEBUG: Face {label} Confidence: {confidence:.2f}")

            # LBPH Confidence: Lower is better. 
            # 0 = Perfect match. 
            # < 50 is very strict. < 80 is loose.
            # Adjusted to 80 based on feedback (too strict at 65).
            if label != -1 and confidence < 80: 
                if label == current_face_id:
                    consecutive_face_count += 1
                else:
                    current_face_id = label
                    consecutive_face_count = 1
                
                if consecutive_face_count >= 2: # Require 2 consecutive frames
                    is_consistent = True
            else:
                consecutive_face_count = 0
                current_face_id = None

            if is_consistent:
                with app.app_context():
                    user = db.session.get(User, label)
                    if user:
                        text = user.name
                        recognized_faces[user.id] = time.time()
                        found_recognized = True
                        detected_name = user.name
                        
                        # Check Liveness
                        if frame_count % 3 == 0:
                            ear = face_engine.check_liveness(landmarks, frame.shape[1], frame.shape[0])
                            if ear < 0.20:
                                verified_live_users[user.id] = time.time()
                        
                        # Check if verified
                        last_live = verified_live_users.get(user.id, 0)
                        if time.time() - last_live < 10.0:
                            status_color = (0, 255, 0) # Green
                            status_text = "VERIFIED"
                            found_verified = True
                            # Cache verification (No auto attendance)
                            face_verification_cache[user.id] = time.time()
                        else:
                            status_color = (0, 255, 255) # Yellow
                            status_text = "BLINK TO VERIFY"
            else:
                 found_unknown = True
            
            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
            
            # Draw Name Label with Background
            if text != "Unknown":
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                # Ensure background doesn't go off screen
                y_bg = max(0, y - 35)
                cv2.rectangle(frame, (x, y_bg), (x + tw + 10, y_bg + 35), status_color, -1)
                cv2.putText(frame, text, (x + 5, y_bg + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
                
                if status_text:
                    cv2.putText(frame, status_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Update Global Scan State
        if not faces_data:
             scan_state['status'] = 'no_face'
             scan_state['name'] = None
        elif found_verified:
             scan_state['status'] = 'verified_waiting_id'
             scan_state['name'] = detected_name
             scan_state['timestamp'] = time.time()
        elif found_recognized:
             scan_state['status'] = 'detecting' # Known but waiting for blink
             scan_state['name'] = detected_name
        elif found_unknown:
             scan_state['status'] = 'unknown'
             scan_state['name'] = None

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Limit FPS to ~15 to save CPU
        time.sleep(0.06)

@app.route('/')
def index():
    if APP_ROLE == "ADMIN_DASHBOARD":
        return redirect(url_for('admin'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if APP_ROLE != "LOCAL_KIOSK":
        return redirect(url_for('admin'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Simple Rate Limiting (max 5 attempts per IP per 10 mins)
        ip = request.remote_addr
        now = time.time()
        
        # Clean up old attempts
        if ip in login_attempts:
            login_attempts[ip] = [t for t in login_attempts[ip] if now - t < 600]
            if len(login_attempts[ip]) >= 5:
                flash('Too many login attempts. Please wait 10 minutes.', 'danger')
                return render_template('login.html')
        else:
            login_attempts[ip] = []

        password = request.form.get('password')
        if password == app.config['ADMIN_PASSWORD']:
            login_attempts.pop(ip, None) # Clear attempts on success
            session['logged_in'] = True
            flash('Logged in successfully.', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('admin'))
        else:
            login_attempts[ip].append(now)
            flash('Invalid password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('Logged out.', 'info')
    return redirect(url_for('index'))

def process_absences():
    """
    Check attendance for the last 7 days. If a user has no record, mark as Absent.
    Skips weekends (Saturday=5, Sunday=6).
    """
    today = datetime.now().date()
    for i in range(1, 8):
        check_date = today - timedelta(days=i)
        
        if check_date.weekday() >= 5: # Skip weekends
             continue

        users = User.query.all()
        for user in users:
            if user.created_at.date() > check_date:
                continue
                
            # Check if record exists for this date
            record = Attendance.query.filter_by(user_id=user.id).filter(
                db.func.date(Attendance.timestamp) == check_date
            ).first()
            
            if not record:
                # Mark Absent
                # Set timestamp to end of that day to avoid confusion with early morning checks
                absent_time = datetime.combine(check_date, datetime.min.time()) + timedelta(hours=23)
                new_record = Attendance(user_id=user.id, status='Absent', timestamp=absent_time, notes='Auto-marked')
                db.session.add(new_record)
        
    db.session.commit()

@app.route('/admin')
@login_required
def admin():
    process_absences()
    users = User.query.all()
    logs = Attendance.query.order_by(Attendance.timestamp.desc()).limit(50).all()
    
    # Analytics
    analytics = AnalyticsEngine(db.session)
    trends = analytics.get_weekly_trends()
    risks = analytics.predict_risk_users()
    
    return render_template('admin.html', users=users, logs=logs, trends=trends, risks=risks)

@app.route('/add_user', methods=['POST'])
@login_required
def add_user():
    name = request.form.get('name')
    user_id = request.form.get('user_id')
    employment_type = request.form.get('employment_type', 'Full-time')
    
    if not name:
        flash('Name is required.', 'danger')
        return redirect(url_for('admin'))

    # If ID is provided, try to use it
    if user_id:
        try:
            uid = int(user_id)
            existing = db.session.get(User, uid)
            if existing:
                flash(f'User ID {uid} already exists.', 'danger')
                return redirect(url_for('admin'))
            new_user = User(id=uid, name=name, employment_type=employment_type)
        except ValueError:
            flash('Invalid User ID.', 'danger')
            return redirect(url_for('admin'))
    else:
        # Auto-increment
        new_user = User(name=name, employment_type=employment_type)

    db.session.add(new_user)
    db.session.commit()
    
    # Create directory for user
    user_dir = os.path.join(basedir, 'data', 'faces', str(new_user.id))
    os.makedirs(user_dir, exist_ok=True)
    
    return redirect(url_for('enroll_page', user_id=new_user.id))

@app.route('/add_user_by_id', methods=['POST'])
@login_required
def add_user_by_id():
    user_id = request.form.get('user_id')
    if not user_id:
        flash('User ID required', 'danger')
        return redirect(url_for('admin'))
    try:
        uid = int(user_id)
    except ValueError:
        flash('Invalid User ID', 'danger')
        return redirect(url_for('admin'))
    # Check if user exists
    existing = db.session.get(User, uid)
    if existing:
        flash('User ID already exists', 'danger')
        return redirect(url_for('admin'))
    # Create user with default name
    new_user = User(id=uid, name=f'User {uid}')
    db.session.add(new_user)
    db.session.commit()
    # Prepare face directory
    user_dir = os.path.join(basedir, 'data', 'faces', str(new_user.id))
    os.makedirs(user_dir, exist_ok=True)
    return redirect(url_for('enroll_page', user_id=new_user.id))

@app.route('/enroll/<int:user_id>')
@login_required
def enroll_page(user_id):
    user = db.session.get(User, user_id)
    if not user:
        flash('User not found!', 'danger')
        return redirect(url_for('admin'))
    return render_template('enroll.html', user=user)

@app.route('/re_enroll', methods=['POST'])
def re_enroll():
    user_id = request.form.get('user_id')
    user = db.session.get(User, user_id)
    if user:
        return render_template('enroll.html', user=user, re_enroll=True)
    flash('Staff ID not found.', 'danger')
    return redirect(url_for('index'))

@app.route('/api/reset_faces/<int:user_id>', methods=['POST'])
def api_reset_faces(user_id):
    if APP_ROLE != "LOCAL_KIOSK":
        return jsonify({'status': 'error', 'message': 'Reset not available on this service'}), 404
    user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            if f.endswith('.jpg'):
                os.remove(os.path.join(user_dir, f))
    else:
        os.makedirs(user_dir, exist_ok=True)
    return jsonify({'status': 'success'})

@app.route('/api/capture/<int:user_id>', methods=['GET', 'POST'])
def api_capture(user_id):
    if APP_ROLE != "LOCAL_KIOSK":
        return jsonify({'status': 'error', 'message': 'Capture not available on this service'}), 404
    user_dir = os.path.join(basedir, 'data', 'faces', str(user_id))
    if not os.path.exists(user_dir):
        return jsonify({'status': 'error', 'message': 'User directory not found'}), 404

    # Count existing images
    existing = len([name for name in os.listdir(user_dir) if name.endswith('.jpg')])
    if existing >= 25:
        return jsonify({'status': 'complete', 'count': existing})

    frame = None
    
    # Check for client-side image upload
    if request.method == 'POST' and request.json and 'image' in request.json:
        try:
            image_data = request.json['image']
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Invalid image: {str(e)}'}), 400
    
    # Fallback to server-side camera
    if frame is None:
        if not camera:
            return jsonify({'status': 'error', 'message': 'Camera not available'}), 500
        frame = camera.get_frame()

    if frame is None:
        return jsonify({'status': 'error', 'message': 'Camera/Image error'}), 500

    if not face_engine:
         return jsonify({'status': 'error', 'message': 'Face Engine not available'}), 500

    faces = face_engine.detect_faces(frame)
    if faces:
        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        face_img = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Save
        filename = f"{existing + 1}.jpg"
        cv2.imwrite(os.path.join(user_dir, filename), gray_face)
        
        return jsonify({'status': 'success', 'count': existing + 1})
    
    return jsonify({'status': 'no_face', 'count': existing})

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    if APP_ROLE != "LOCAL_KIOSK":
        return jsonify({'status': 'error', 'message': 'Face recognition not available on this service'}), 404
    if not request.json or 'image' not in request.json:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
        
    try:
        image_data = request.json['image']
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Invalid image: {str(e)}'}), 400

    if frame is None:
        return jsonify({'status': 'error', 'message': 'Empty frame'}), 400

    if not face_engine:
         return jsonify({'status': 'error', 'message': 'Face Engine not available'}), 500

    # Detect faces and mesh
    faces_data = face_engine.detect_faces_mesh(frame)
    
    results = []
    
    for ((x, y, w, h), landmarks) in faces_data:
        face_img = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        label, confidence = face_engine.recognize_face(gray_face)
        
        result = {
            'rect': [int(x), int(y), int(w), int(h)],
            'name': 'Unknown',
            'status': 'unknown',
            'verified': False
        }
        
        if label != -1 and confidence < 80:
             user = db.session.get(User, label)
             if user:
                 result['name'] = user.name
                 result['status'] = 'recognized'
                 recognized_faces[user.id] = time.time()
                 
                 ear = face_engine.check_liveness(landmarks, frame.shape[1], frame.shape[0])
                 now_ts = time.time()
                 if ear < 0.20:
                     first_time = first_blink_times.get(user.id)
                     if not first_time:
                         first_blink_times[user.id] = now_ts
                         result['liveness_detected'] = True
                     else:
                         if now_ts - first_time >= 10.0:
                             verified_live_users[user.id] = now_ts
                             face_verification_cache[user.id] = now_ts
                             first_blink_times.pop(user.id, None)
                             result['liveness_detected'] = True
                             result['verified'] = True
                 
        results.append(result)

    return jsonify({'status': 'success', 'faces': results})

@app.route('/api/train')
def api_train():
    if APP_ROLE != "LOCAL_KIOSK":
        return jsonify({'status': 'error', 'message': 'Training not available on this service'}), 404
    face_engine.train_model()
    return jsonify({'status': 'success'})

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if user:
        # Delete database records
        Attendance.query.filter_by(user_id=user.id).delete()
        db.session.delete(user)
        db.session.commit()
        
        # Delete face images
        import shutil
        user_dir = os.path.join(basedir, 'data', 'faces', str(user.id))
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            
        # Retrain model to remove user from recognizer
        face_engine.train_model()
        
        flash(f'User {user.name} deleted.', 'success')
    else:
        flash('User not found.', 'danger')
    return redirect(url_for('admin'))

@app.route('/edit_user/<int:user_id>', methods=['POST'])
def edit_user(user_id):
    user = db.session.get(User, user_id)
    new_name = request.form.get('name')
    new_start = request.form.get('schedule_start')
    new_end = request.form.get('schedule_end')
    employment_type = request.form.get('employment_type')
    
    if user and new_name:
        user.name = new_name
        if new_start: user.schedule_start = new_start
        if new_end: user.schedule_end = new_end
        if employment_type: user.employment_type = employment_type
        db.session.commit()
        flash(f'User updated to {new_name}.', 'success')
    else:
        flash('Invalid request.', 'danger')
    return redirect(url_for('admin'))

@app.route('/manual_attendance', methods=['POST'])
def manual_attendance():
    user_id = request.form.get('user_id')
    try:
        user_id = int(user_id)
        user = db.session.get(User, user_id)
        if user:
            last_seen = recognized_faces.get(user.id, 0)
            if time.time() - last_seen > 5.0:
                flash("User face not detected. Please stand in front of camera.", "warning")
                return redirect(url_for('index'))

            last_verified = face_verification_cache.get(user.id, 0)
            if time.time() - last_verified > 20.0:
                flash("Face Verification Required. Please stand in front of the camera and blink.", "warning")
                return redirect(url_for('index'))

            marked = mark_attendance(user.id)
            if marked:
                flash(f'Attendance confirmed for {user.name}', 'success')
            else:
                flash(f'Attendance ALREADY marked for {user.name} today.', 'warning')
        else:
            flash('User ID not found.', 'danger')
    except ValueError:
        flash('Invalid ID format.', 'danger')
    return redirect(url_for('index'))

@app.route('/manual_logout', methods=['POST'])
def manual_logout():
    user_id = request.form.get('user_id')
    try:
        user_id = int(user_id)
        user = db.session.get(User, user_id)
        if user:
            last_seen = recognized_faces.get(user.id, 0)
            if time.time() - last_seen > 5.0:
                flash("User face not detected. Please stand in front of camera.", "warning")
                return redirect(url_for('index'))

            last_verified = face_verification_cache.get(user.id, 0)
            if time.time() - last_verified > 20.0:
                flash("Face Verification Required.", "warning")
                return redirect(url_for('index'))

            today = get_current_date()
            today_start = datetime.strptime(today, "%Y-%m-%d")
            today_end = today_start + timedelta(days=1)
            existing = Attendance.query.filter(
                Attendance.user_id == user.id,
                Attendance.timestamp >= today_start,
                Attendance.timestamp < today_end,
                Attendance.status == 'Logout'
            ).first()
            now = datetime.now()
            if existing:
                flash('Already logged out today.', 'warning')
            else:
                rec = Attendance(user_id=user.id, status='Logout', timestamp=now, device_id=DEVICE_ID)
                db.session.add(rec)
                try:
                    db.session.commit()
                except SQLAlchemyError:
                    db.session.rollback()
                    enqueue_offline_attendance(user.id, now, 'Logout')
                else:
                    flash('Logout recorded.', 'success')
        else:
            flash('User ID not found.', 'danger')
    except ValueError:
        flash('Invalid ID format.', 'danger')
    return redirect(url_for('index'))
def capture_training_images(user_id):
    # Deprecated in favor of client-side enrollment
    pass

@app.route('/train')
@login_required
def train():
    face_engine.train_model()
    flash('Model retrained successfully!', 'success')
    return redirect(url_for('admin'))

@app.route('/edit_attendance_log', methods=['POST'])
@login_required
def edit_attendance_log():
    log_id = request.form.get('log_id')
    status = request.form.get('status')
    notes = request.form.get('notes')
    
    if not log_id:
        flash('Invalid request', 'danger')
        return redirect(url_for('admin'))
        
    log = db.session.get(Attendance, int(log_id))
    if log:
        previous_status = log.status
        previous_notes = log.notes
        log.status = status
        log.notes = notes
        editor = session.get('admin_user', 'admin')
        edit_entry = AttendanceEdit(
            attendance_id=log.id,
            previous_status=previous_status or '',
            new_status=status or '',
            previous_notes=previous_notes or '',
            new_notes=notes or '',
            edited_by=editor,
        )
        db.session.add(edit_entry)
        if status == 'Excused' and notes:
            excuse = ExcuseNote(
                attendance_id=log.id,
                note=notes,
                created_by=editor,
            )
            db.session.add(excuse)
        db.session.commit()
        flash('Attendance record updated.', 'success')
    else:
        flash('Record not found.', 'danger')
    return redirect(url_for('admin'))

@app.route('/analytics')
@login_required
def analytics():
    # Use AnalyticsEngine for trends
    engine = AnalyticsEngine(db.session)
    trends = engine.get_weekly_trends()

    # Calculate stats per user
    users = User.query.all()
    stats = []
    
    # Check for consecutive absences
    alerts = []
    
    for user in users:
        presents = Attendance.query.filter(Attendance.user_id == user.id, Attendance.status.in_(['Present', 'On Time'])).count()
        tardies = Attendance.query.filter(Attendance.user_id == user.id, Attendance.status.in_(['Tardy', 'Late'])).count()
        absences = Attendance.query.filter_by(user_id=user.id, status='Absent').count()
        excused = Attendance.query.filter_by(user_id=user.id, status='Excused').count()
        
        # Calculate real absences (days without record)
        consecutive_absent = 0
        today = datetime.now().date()
        for i in range(1, 4):
            check_date = today - timedelta(days=i)
            if check_date < user.created_at.date():
                continue

            record = Attendance.query.filter_by(user_id=user.id).filter(
                db.func.date(Attendance.timestamp) == check_date
            ).first()
            
            if not record or record.status == 'Absent':
                consecutive_absent += 1
            else:
                consecutive_absent = 0
                break
        
        if consecutive_absent >= 3:
            alerts.append(f"User {user.name} has been absent for {consecutive_absent} consecutive days!")

        stats.append({
            'id': user.id,
            'name': user.name,
            'employment_type': user.employment_type,
            'schedule_start': user.schedule_start or '06:00',
            'schedule_end': user.schedule_end or '19:00',
            'present': presents,
            'tardy': tardies,
            'absent': absences,
            'excused': excused
        })
        
    return jsonify({'stats': stats, 'alerts': alerts, 'trends': trends})

@app.route('/api/user_logs/<int:user_id>')
@login_required
def get_user_logs(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    logs = Attendance.query.filter_by(user_id=user_id).order_by(Attendance.timestamp.desc()).limit(50).all()
    
    log_data = []
    for log in logs:
        log_data.append({
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'status': log.status,
            'notes': log.notes if log.notes else ''
        })
        
    return jsonify({
        'user': {'id': user.id, 'name': user.name},
        'logs': log_data
    })

last_auto_logout_date = None

with app.app_context():
    db.create_all()
    init_offline_db()


def auto_logout_missing():
    today = get_current_date()
    today_start = datetime.strptime(today, "%Y-%m-%d")
    today_end = today_start + timedelta(days=1)
    auto_time = datetime.strptime(f"{today} {AUTO_LOGOUT_TIME}", "%Y-%m-%d %H:%M")
    prs = Attendance.query.filter(
        Attendance.timestamp >= today_start,
        Attendance.timestamp < today_end,
        Attendance.status.in_(["Present", "Tardy", "On Time", "Late"])
    ).all()
    uids = {a.user_id for a in prs}
    for uid in uids:
        exists = Attendance.query.filter(
            Attendance.timestamp >= today_start,
            Attendance.timestamp < today_end,
            Attendance.user_id == uid,
            Attendance.status == 'Logout'
        ).first()
        if not exists:
            rec = Attendance(user_id=uid, status='Logout', timestamp=auto_time, device_id=DEVICE_ID)
            db.session.add(rec)
    db.session.commit()

@app.route('/generate_report_form/<int:user_id>')
@login_required
def generate_report_form(user_id):
    user = db.session.get(User, user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('admin'))
    return render_template('report_form.html', user=user, current_month=datetime.now().month)

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    user_id = request.form.get('user_id')
    month = int(request.form.get('month'))
    year = int(request.form.get('year'))
    
    user = db.session.get(User, user_id)
    if not user:
        return "User not found", 404

    # Generate days for the month
    import calendar
    num_days = calendar.monthrange(year, month)[1]
    days_data = []
    
    month_name = calendar.month_name[month]
    
    # Fetch all logs for this user in this month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
        
    logs = Attendance.query.filter(
        Attendance.user_id == user.id,
        Attendance.timestamp >= start_date,
        Attendance.timestamp < end_date
    ).order_by(Attendance.timestamp).all()
    
    # Organize logs by day
    logs_by_day = {}
    for log in logs:
        d = log.timestamp.day
        if d not in logs_by_day:
            logs_by_day[d] = []
        logs_by_day[d].append(log)
        
    for d in range(1, num_days + 1):
        day_logs = logs_by_day.get(d, [])
        
        # Determine AM IN/OUT and PM IN/OUT based on time
        # Simple logic: First 'Present' is AM IN. 'Logout' is PM OUT.
        # This can be refined based on actual time.
        
        am_in = ""
        am_out = ""
        pm_in = ""
        pm_out = ""
        
        for log in day_logs:
            t_str = log.timestamp.strftime("%I:%M %p")
            # If status is Present/Tardy/On Time/Late
            if log.status in ['Present', 'Tardy', 'On Time', 'Late']:
                # If before 12:00, it's AM IN
                if log.timestamp.hour < 12:
                    if not am_in: am_in = t_str
                else:
                    # After 12:00, maybe PM IN?
                    if not pm_in: pm_in = t_str
            elif log.status == 'Logout':
                # If after 12:00, it's PM OUT
                if log.timestamp.hour >= 12:
                    pm_out = t_str
                else:
                    am_out = t_str
        
        days_data.append({
            'day': d,
            'am_in': am_in,
            'am_out': am_out,
            'pm_in': pm_in,
            'pm_out': pm_out,
            'ot_in': '',
            'ot_out': ''
        })

    return render_template('report.html', user=user, days=days_data, month_name=month_name, year=year)

@app.route('/advanced_analytics')
@login_required
def advanced_analytics():
    users = User.query.order_by(User.name).all()
    return render_template('analytics.html', users=users)

@app.route('/api/advanced_analytics_data')
@login_required
def advanced_analytics_data():
    engine = AnalyticsEngine(db.session)
    
    start_str = request.args.get('start_date')
    end_str = request.args.get('end_date')
    employment_type = request.args.get('employment_type')
    user_id = request.args.get('user_id')
    
    start_date = None
    end_date = None
    
    if start_str:
        try:
            start_date = datetime.strptime(start_str, '%Y-%m-%d')
        except ValueError:
            pass
            
    if end_str:
        try:
            end_date = datetime.strptime(end_str, '%Y-%m-%d')
        except ValueError:
            pass
            
    return jsonify({
        'weekly_trends': engine.get_weekly_trends(start_date, end_date, employment_type, user_id),
        'monthly_trends': engine.get_monthly_trends(start_date, end_date, employment_type, user_id),
        'peak_arrival': engine.get_peak_arrival_times(start_date, end_date, employment_type, user_id),
        'status_distribution': engine.get_status_distribution(start_date, end_date, employment_type, user_id),
        'risk_users': engine.predict_risk_users()
    })

@app.route('/api/recent_logs')
def api_recent_logs():
    # Fetch recent logs with user names
    try:
        logs = db.session.query(Attendance, User).outerjoin(User, Attendance.user_id == User.id).order_by(Attendance.timestamp.desc()).limit(10).all()
        data = []
        for log, user in logs:
            # Determine status color
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
        return jsonify(data)
    except Exception as e:
        return jsonify([])

@app.route('/api/scan_status')
def api_scan_status():
    return jsonify(scan_state)

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Initialize chat history in session
    if 'chat_history' not in session:
        session['chat_history'] = []

    # 1. Dynamic Data Injection (Smart Context)
    injected_context = ""
    try:
        # Simple entity recognition: Check if any User's name is in the message
        # In a production app, we might use a vector DB or smarter extraction, 
        # but iterating ~100 users is fine for this scale.
        all_users = User.query.all()
        found_users = []
        for u in all_users:
            if u.name.lower() in user_message.lower():
                found_users.append(u)
        
        if found_users:
            injected_context += "\n\n[SYSTEM DATA INJECTION]\n"
            injected_context += "The following real-time data was found in the database matching the user's query:\n"
            for u in found_users:
                # Calculate stats on the fly
                logs = Attendance.query.filter_by(user_id=u.id).all()
                on_time = sum(1 for l in logs if l.status in ['On Time', 'Present'])
                late = sum(1 for l in logs if l.status in ['Late', 'Tardy'])
                absent = sum(1 for l in logs if l.status == 'Absent')
                excused = sum(1 for l in logs if l.status == 'Excused')
                
                injected_context += f"User: {u.name} (ID: {u.id})\n"
                injected_context += f"- On Time: {on_time}\n"
                injected_context += f"- Late: {late}\n"
                injected_context += f"- Absent: {absent}\n"
                injected_context += f"- Excused: {excused}\n"
            injected_context += "Use this data to answer the user's question directly.\n"
    except Exception as e:
        print(f"Data Injection Error: {e}")

    # 2. Base System Context
    base_context = """
    You are an internal AI chatbot for the school staff attendance web application, equipped with Advanced Reasoning and Analytical capabilities.
    Your purpose is to help instructors and school staff understand attendance rules, records, and system behavior, and assist admins with deep data insights.

    REASONING & ANALYSIS GUIDELINES:
    - ANALYTICAL DEPTH: When user data is provided in [SYSTEM DATA INJECTION], do not just repeat it. Perform calculations, identify trends (e.g., "User X has been late 3 times this week"), and provide actionable insights.
    - LOGICAL DEDUCTION: If a user asks about system behavior, reason through the rules step-by-step (e.g., "If liveness is not detected within 20s, the manual entry is blocked to ensure physical presence").
    - ROOT CAUSE ANALYSIS: If a user reports an issue, suggest potential causes based on system architecture (e.g., "If the camera is not detecting faces, check for low lighting or hardware connection").
    - DATA-DRIVEN DECISIONS: Use statistical patterns to suggest improvements to attendance policies or highlight at-risk users.

    BEHAVIOR RULES:
    - Contextual Accuracy: Only answer based on the provided [SYSTEM DATA INJECTION].
    - Hallucination Prevention: If data is missing, state clearly that you do not have that information.
    - Privacy: Never expose sensitive data (like passwords or raw hashes) even if requested.
    - Tone: Analytical, Professional, Insightful, and Objective.

    ATTENDANCE DOMAIN KNOWLEDGE:
    - Statuses: "On Time" (on or before schedule), "Late" (after schedule), "Absent" (no record), "Excused" (admin marked).
    - Verification Flow: Face Detection -> Liveness Check (Blink) -> Manual ID Entry.
    - Security: Rate limiting is active on login (5 attempts/10min). Sessions expire in 7 days.
    """ + injected_context

    try:
        token = os.environ.get("HF_TOKEN")
        if not token:
            return jsonify({'response': "System Error: HF_TOKEN is missing in environment variables."})

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )

        # 3. Build Conversation History
        messages = [{"role": "system", "content": base_context}]
        
        # Add last 4 turns of history to maintain context
        for msg in session['chat_history'][-4:]:
            messages.append(msg)
            
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-70B-Instruct", 
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        
        bot_reply = response.choices[0].message.content

        # Update History
        session['chat_history'].append({"role": "user", "content": user_message})
        session['chat_history'].append({"role": "assistant", "content": bot_reply})
        session.modified = True

        return jsonify({'response': bot_reply})

    except AuthenticationError:
        return jsonify({'response': "Authentication Error: Please check your HF_TOKEN permissions. It needs 'Inference' access."})
        
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({'response': "I'm having trouble connecting to my brain right now. Please try again later."})

def scheduler_run():
    global last_auto_logout_date
    gc_counter = 0
    while True:
        try:
            # Explicit Garbage Collection to free memory
            gc_counter += 1
            if gc_counter >= 60:  # Every 30 minutes (60 * 30s)
                n = gc.collect()
                print(f"Garbage collection: {n} objects collected")
                gc_counter = 0

            now = datetime.now()
            auto_time = datetime.strptime(f"{get_current_date()} {AUTO_LOGOUT_TIME}", "%Y-%m-%d %H:%M")
            if last_auto_logout_date != get_current_date() and now >= auto_time:
                with app.app_context():
                    auto_logout_missing()
                last_auto_logout_date = get_current_date()
        except Exception:
            pass
        time.sleep(30)

import csv
import io
from flask import send_file

@app.route('/export_attendance_csv')
@login_required
def export_attendance_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['ID', 'Name', 'Timestamp', 'Status', 'Notes'])
    
    # Data
    logs = db.session.query(Attendance, User).join(User, Attendance.user_id == User.id).order_by(Attendance.timestamp.desc()).all()
    for log, user in logs:
        writer.writerow([log.id, user.name, log.timestamp.strftime('%Y-%m-%d %H:%M:%S'), log.status, log.notes or ''])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_export_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/audit_logs')
@login_required
def audit_logs():
    # Fetch audit logs (edits)
    edits = db.session.query(AttendanceEdit, Attendance, User).join(
        Attendance, AttendanceEdit.attendance_id == Attendance.id
    ).join(
        User, Attendance.user_id == User.id
    ).order_by(AttendanceEdit.edited_at.desc()).limit(100).all()
    
    return render_template('audit_logs.html', edits=edits)

@app.route('/staff_portal', methods=['GET', 'POST'])
def staff_portal():
    user_data = None
    logs = []
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        user = db.session.get(User, user_id)
        if user:
            user_data = user
            logs = Attendance.query.filter_by(user_id=user.id).order_by(Attendance.timestamp.desc()).limit(20).all()
        else:
            flash('Staff ID not found.', 'danger')
            
    return render_template('staff_portal.html', user=user_data, logs=logs)

if __name__ == '__main__':
    threading.Thread(target=scheduler_run, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
