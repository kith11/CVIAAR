from __future__ import annotations

from datetime import datetime

AM_SESSION = "AM"
PM_SESSION = "PM"
MIDDAY_HOUR = 12

DEFAULT_LATE_GRACE_MINUTES = 15
DEFAULT_WORK_DAYS = (0, 1, 2, 3, 4)  # Mon-Fri (Python weekday ints)

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


def parse_hhmm(value: str | None, default: tuple[int, int] = (6, 0)) -> tuple[int, int]:
    """Parse an 'HH:MM' string into an (hour, minute) tuple, falling back on errors."""
    try:
        hour, minute = str(value).split(":")
        return int(hour), int(minute)
    except (ValueError, AttributeError):
        return default


def is_working_day(value, work_days=DEFAULT_WORK_DAYS) -> bool:
    """Whether the given date/datetime falls on a configured working weekday."""
    return value.weekday() in set(work_days)


def classify_login_status(
    timestamp: datetime,
    schedule_start: str | None,
    session: str | None = None,
    grace_minutes: int = DEFAULT_LATE_GRACE_MINUTES,
) -> str:
    """Classify a login as 'On Time' or 'Late', respecting the AM/PM session.

    Lateness is measured against the scheduled start of the relevant session.
    A staffer scheduled in the morning who checks in again in the afternoon
    (e.g. returning from lunch) is treated as present rather than late, since
    there is no separate PM schedule to be late against.
    """
    resolved = normalize_session(session, timestamp)
    sched_hour, sched_minute = parse_hhmm(schedule_start)

    if resolved == PM_SESSION and sched_hour < MIDDAY_HOUR:
        return "On Time"

    scheduled_minutes = sched_hour * 60 + sched_minute
    actual_minutes = timestamp.hour * 60 + timestamp.minute
    return "On Time" if actual_minutes <= scheduled_minutes + grace_minutes else "Late"
