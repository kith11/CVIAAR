from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, inspect, text
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """
    Represents a user in the system.

    Attributes:
        id (int): Primary key.
        name (str): The user's full name.
        email (str): The user's email address (used for notifications).
        staff_code (str): A unique 6-digit code for staff portal access.
        schedule_start (str): The user's expected start time (HH:MM).
        schedule_end (str): The user's expected end time (HH:MM).
        employment_type (str): The user's employment status (e.g., Full-time, Part-time).
        role (str): The user's role (e.g., staff, admin).
    """
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
    """
    Represents a single attendance record.

    Attributes:
        id (int): Primary key.
        sync_key (str): A unique UUID for offline synchronization.
        user_id (int): Foreign key to the User model.
        timestamp (datetime): The date and time of the attendance event.
        status (str): The status of the attendance (e.g., On Time, Late, Logout).
        notes (str): Optional notes for the record.
        device_id (str): The ID of the device that recorded the attendance.
        synced (int): A flag indicating if the record has been synced with a remote server.
    """
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True)
    sync_key = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(String(20), nullable=False)
    session = Column(String(2), nullable=True)
    event_type = Column(String(20), nullable=True)
    auto_generated = Column(Integer, default=0)
    notes = Column(String(200), nullable=True)
    device_id = Column(String(50), nullable=True)
    synced = Column(Integer, default=0) # 0: pending, 1: synced
    synced_at = Column(DateTime, nullable=True)
    user = relationship("User", back_populates="attendances")


def ensure_attendance_schema(engine) -> None:
    """Adds newer attendance columns to existing databases when needed."""
    inspector = inspect(engine)
    try:
        existing_columns = {column["name"] for column in inspector.get_columns("attendance_logs")}
    except Exception:
        return

    statements = []
    if "session" not in existing_columns:
        statements.append("ALTER TABLE attendance_logs ADD COLUMN session VARCHAR(2)")
    if "event_type" not in existing_columns:
        statements.append("ALTER TABLE attendance_logs ADD COLUMN event_type VARCHAR(20)")
    if "auto_generated" not in existing_columns:
        statements.append("ALTER TABLE attendance_logs ADD COLUMN auto_generated INTEGER DEFAULT 0")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


class AttendanceEdit(Base):
    """
    Logs any modifications made to an attendance record.

    This table provides an audit trail for changes to attendance data.

    Attributes:
        id (int): Primary key.
        attendance_id (int): Foreign key to the Attendance model.
        previous_status (str): The original status of the attendance record.
        new_status (str): The updated status.
        edited_by (str): The user who made the edit.
    """
    __tablename__ = "attendance_edits"

    id = Column(Integer, primary_key=True)
    attendance_id = Column(Integer, ForeignKey("attendance_logs.id"), nullable=False)
    previous_status = Column(String(20), nullable=False)
    new_status = Column(String(20), nullable=False)
    previous_notes = Column(String(200), nullable=True)
    new_notes = Column(String(200), nullable=True)
    edited_by = Column(String(100), nullable=True)
    edited_at = Column(DateTime, default=datetime.now)


class AuditEvent(Base):
    """Generic admin/system audit event."""

    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True)
    action_type = Column(String(50), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(50), nullable=True)
    actor = Column(String(100), nullable=True)
    summary = Column(String(200), nullable=False)
    details = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)


class ExcuseNote(Base):
    """
    Represents an excuse note submitted for a specific attendance record.

    Attributes:
        id (int): Primary key.
        attendance_id (int): Foreign key to the Attendance model.
        note (str): The content of the excuse note.
        created_by (str): The user who submitted the note.
    """
    __tablename__ = "excuse_notes"

    id = Column(Integer, primary_key=True)
    attendance_id = Column(Integer, ForeignKey("attendance_logs.id"), nullable=False)
    note = Column(String(500), nullable=False)
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class Device(Base):
    """
    Represents a physical device (kiosk) in the system.

    Attributes:
        id (int): Primary key.
        device_id (str): A unique identifier for the device.
        name (str): A human-readable name for the device.
        location (str): The physical location of the device.
    """
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True)
    device_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=True)
    location = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
