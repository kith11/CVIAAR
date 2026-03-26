from __future__ import annotations

from datetime import datetime

AM_SESSION = "AM"
PM_SESSION = "PM"
MIDDAY_HOUR = 12

ON_TIME_STATUSES = {"Present", "On Time"}
LATE_STATUSES = {"Late", "Tardy"}
LOGIN_STATUSES = ON_TIME_STATUSES | LATE_STATUSES
LOGOUT_STATUSES = {"Logout"}
ABSENT_STATUSES = {"Absent", "AM Absent", "PM Absent"}


def resolve_session(timestamp: datetime | None) -> str | None:
    if timestamp is None:
        return None
    return AM_SESSION if timestamp.hour < MIDDAY_HOUR else PM_SESSION


def normalize_session(session: str | None, timestamp: datetime | None = None) -> str | None:
    if session in {AM_SESSION, PM_SESSION}:
        return session
    return resolve_session(timestamp)


def session_absent_status(session: str) -> str:
    return "AM Absent" if session == AM_SESSION else "PM Absent"


def infer_event_type(status: str | None, event_type: str | None = None) -> str | None:
    if event_type in {"login", "logout", "auto_absent"}:
        return event_type
    if status in LOGIN_STATUSES:
        return "login"
    if status in LOGOUT_STATUSES:
        return "logout"
    if status in ABSENT_STATUSES:
        return "auto_absent"
    return None


def is_present_status(status: str | None) -> bool:
    return status in LOGIN_STATUSES


def is_on_time_status(status: str | None) -> bool:
    return status in ON_TIME_STATUSES


def is_late_status(status: str | None) -> bool:
    return status in LATE_STATUSES


def is_absent_status(status: str | None) -> bool:
    return status in ABSENT_STATUSES


def is_login_record(status: str | None, event_type: str | None = None) -> bool:
    return infer_event_type(status, event_type) == "login"


def is_logout_record(status: str | None, event_type: str | None = None) -> bool:
    return infer_event_type(status, event_type) == "logout"
