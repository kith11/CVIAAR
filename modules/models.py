from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), nullable=True)
    staff_code = Column(String(6), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    schedule_start = Column(String(5), default="06:00")
    schedule_end = Column(String(5), default="19:00")
    employment_type = Column(String(20), default="Full-time")
    role = Column(String(20), default="staff")  # roles: staff, admin
    last_report_sent = Column(DateTime, nullable=True)
    attendances = relationship("Attendance", back_populates="user", lazy=True)


class Attendance(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True)
    sync_key = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(String(20), nullable=False)
    notes = Column(String(200), nullable=True)
    device_id = Column(String(50), nullable=True)
    synced_at = Column(DateTime, nullable=True)
    user = relationship("User", back_populates="attendances")


class AttendanceEdit(Base):
    __tablename__ = "attendance_edits"

    id = Column(Integer, primary_key=True)
    attendance_id = Column(Integer, ForeignKey("attendance_logs.id"), nullable=False)
    previous_status = Column(String(20), nullable=False)
    new_status = Column(String(20), nullable=False)
    previous_notes = Column(String(200), nullable=True)
    new_notes = Column(String(200), nullable=True)
    edited_by = Column(String(100), nullable=True)
    edited_at = Column(DateTime, default=datetime.now)


class ExcuseNote(Base):
    __tablename__ = "excuse_notes"

    id = Column(Integer, primary_key=True)
    attendance_id = Column(Integer, ForeignKey("attendance_logs.id"), nullable=False)
    note = Column(String(500), nullable=False)
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True)
    device_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=True)
    location = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
