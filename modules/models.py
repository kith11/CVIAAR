from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    schedule_start = db.Column(db.String(5), default="06:00")
    schedule_end = db.Column(db.String(5), default="19:00")
    employment_type = db.Column(db.String(20), default="Full-time")
    role = db.Column(db.String(20), default="staff")  # roles: staff, admin
    attendances = db.relationship("Attendance", backref="user", lazy=True)


class Attendance(db.Model):
    __tablename__ = "attendance_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    status = db.Column(db.String(20), nullable=False)
    notes = db.Column(db.String(200), nullable=True)
    device_id = db.Column(db.String(50), nullable=True)


class AttendanceEdit(db.Model):
    __tablename__ = "attendance_edits"

    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey("attendance_logs.id"), nullable=False)
    previous_status = db.Column(db.String(20), nullable=False)
    new_status = db.Column(db.String(20), nullable=False)
    previous_notes = db.Column(db.String(200), nullable=True)
    new_notes = db.Column(db.String(200), nullable=True)
    edited_by = db.Column(db.String(100), nullable=True)
    edited_at = db.Column(db.DateTime, default=datetime.now)


class ExcuseNote(db.Model):
    __tablename__ = "excuse_notes"

    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey("attendance_logs.id"), nullable=False)
    note = db.Column(db.String(500), nullable=False)
    created_by = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)


class Device(db.Model):
    __tablename__ = "devices"

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
