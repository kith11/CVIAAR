#!/usr/bin/env python3
"""
Seed realistic dummy attendance + analytics data.

Run inside Docker (recommended):
  docker compose run --rm web python scripts/seed_dummy_data.py --reset --days 60
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta

# Avoid initializing camera/face pipeline when importing app
os.environ.setdefault("APP_ROLE", "ADMIN_DASHBOARD")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from app import SessionLocal
from modules.models import Attendance, AttendanceEdit, ExcuseNote, User


def _dt_at(date: datetime, hhmm: str) -> datetime:
    hh, mm = hhmm.split(":")
    return datetime(date.year, date.month, date.day, int(hh), int(mm), 0)


def seed(days: int, reset: bool, seed_value: int, device_id: str) -> None:
    rng = random.Random(seed_value)
    db = SessionLocal()

    try:
        users = db.query(User).order_by(User.id.asc()).all()
        if not users:
            print("No users found. Seed users first.")
            return

        if reset:
            db.query(AttendanceEdit).delete()
            db.query(ExcuseNote).delete()
            db.query(Attendance).delete()
            db.commit()

        # Make sure there's enough historical coverage for analytics
        today = datetime.now()
        start_day = today - timedelta(days=days)

        created_att = 0
        created_edits = 0
        created_excuses = 0

        # Pick a few "risk" users (more late/absent) to populate the risk table
        risk_ids = {u.id for u in rng.sample(users, k=min(3, len(users)))}
        
        # New: Define behavioral personas for more realistic patterns
        # 1. "The Early Bird" - always 15-30 mins early
        # 2. "The Consistent Professional" - arrives +/- 5 mins of schedule
        # 3. "The Late Starter" - consistently 5-15 mins late
        # 4. "The Session Hopper" - multiple logins/logouts per day
        personas = {}
        for i, u in enumerate(users):
            if i % 4 == 0: personas[u.id] = "early_bird"
            elif i % 4 == 1: personas[u.id] = "consistent"
            elif i % 4 == 2: personas[u.id] = "late_starter"
            else: personas[u.id] = "hopper"

        for d in range(days):
            day = start_day + timedelta(days=d)
            # Mostly weekdays, but some weekend activity for "hoppers"
            is_weekend = day.weekday() >= 5
            if is_weekend and rng.random() < 0.7:
                continue

            for u in users:
                persona = personas.get(u.id, "consistent")
                
                # Weekend skip for non-hoppers
                if is_weekend and persona != "hopper" and rng.random() < 0.95:
                    continue

                # Skip if already has an arrival record that day (idempotent-ish)
                day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
                day_end = day_start + timedelta(days=1)
                existing = (
                    db.query(Attendance).filter(Attendance.user_id == u.id)
                    .filter(Attendance.timestamp >= day_start, Attendance.timestamp < day_end)
                    .first()
                )
                if existing and not reset:
                    continue

                sched_start = u.schedule_start or "08:00"
                sched_end = u.schedule_end or "17:00"

                # Arrival logic based on persona
                if persona == "early_bird":
                    offset = rng.randint(-30, -10)
                    status = "On Time"
                elif persona == "late_starter":
                    offset = rng.randint(5, 20)
                    status = "Late"
                elif persona == "hopper":
                    offset = rng.randint(-15, 15)
                    status = "On Time" if offset <= 0 else "Late"
                else: # consistent
                    offset = rng.randint(-5, 5)
                    status = "On Time"

                # Override for risk users
                if u.id in risk_ids and rng.random() < 0.4:
                    offset = rng.randint(30, 180)
                    status = "Late"

                # Random absenteeism
                if rng.random() < (0.15 if u.id in risk_ids else 0.02):
                    continue

                ts = _dt_at(day, sched_start) + timedelta(minutes=offset)
                rec = Attendance(
                    user_id=u.id,
                    timestamp=ts,
                    status=status,
                    notes=f"Persona: {persona}",
                    device_id=device_id,
                )
                db.add(rec)
                created_att += 1

                # Hopper persona: Multiple sessions
                if persona == "hopper" and rng.random() < 0.6:
                    # Mid-day logout/login
                    mid_logout = ts + timedelta(hours=rng.randint(2, 4))
                    mid_login = mid_logout + timedelta(minutes=rng.randint(30, 90))
                    db.add(Attendance(user_id=u.id, timestamp=mid_logout, status="Logout", device_id=device_id))
                    db.add(Attendance(user_id=u.id, timestamp=mid_login, status="Present", device_id=device_id))
                    created_att += 2

                # Final Logout
                logout_offset = rng.randint(-10, 45)
                logout_ts = _dt_at(day, sched_end) + timedelta(minutes=logout_offset)
                db.add(Attendance(user_id=u.id, timestamp=logout_ts, status="Logout", device_id=device_id))
                created_att += 1

        db.commit()

        print("Seed complete.")
        print(f"Users: {len(users)}")
        print(f"Attendance records created: {created_att}")
        print(f"Attendance edits created: {created_edits}")
        print(f"Excuse notes created: {created_excuses}")
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--reset", action="store_true", help="Delete existing attendance/edit/excuse data first")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device-id", type=str, default=os.getenv("DEVICE_ID", "seed-device"))
    args = ap.parse_args()

    if args.days < 1:
        print("--days must be >= 1")
        raise SystemExit(2)

    seed(days=args.days, reset=args.reset, seed_value=args.seed, device_id=args.device_id)


if __name__ == "__main__":
    main()

