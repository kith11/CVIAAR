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

from app import app, db  # noqa: E402
from modules.models import Attendance, AttendanceEdit, ExcuseNote, User  # noqa: E402


def _dt_at(date: datetime, hhmm: str) -> datetime:
    hh, mm = hhmm.split(":")
    return datetime(date.year, date.month, date.day, int(hh), int(mm), 0)


def seed(days: int, reset: bool, seed_value: int, device_id: str) -> None:
    rng = random.Random(seed_value)

    with app.app_context():
        users = User.query.order_by(User.id.asc()).all()
        if not users:
            print("No users found. Seed users first.")
            return

        if reset:
            AttendanceEdit.query.delete()
            ExcuseNote.query.delete()
            Attendance.query.delete()
            db.session.commit()

        # Make sure there's enough historical coverage for analytics
        today = datetime.now()
        start_day = today - timedelta(days=days)

        created_att = 0
        created_edits = 0
        created_excuses = 0

        # Pick a few "risk" users (more late/absent) to populate the risk table
        risk_ids = {u.id for u in rng.sample(users, k=min(3, len(users)))}

        for d in range(days):
            day = start_day + timedelta(days=d)
            # Mostly weekdays
            if day.weekday() >= 5 and rng.random() < 0.85:
                continue

            for u in users:
                # Skip if already has an arrival record that day (idempotent-ish)
                day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
                day_end = day_start + timedelta(days=1)
                existing = (
                    Attendance.query.filter(Attendance.user_id == u.id)
                    .filter(Attendance.timestamp >= day_start, Attendance.timestamp < day_end)
                    .first()
                )
                if existing and not reset:
                    continue

                sched_start = u.schedule_start or "06:00"

                # Rates
                base_absent = 0.04
                base_excused = 0.03
                base_late = 0.14
                if u.id in risk_ids:
                    base_absent = 0.12
                    base_late = 0.35

                roll = rng.random()

                if roll < base_absent:
                    ts = _dt_at(day, sched_start) + timedelta(minutes=rng.randint(0, 30))
                    rec = Attendance(
                        user_id=u.id,
                        timestamp=ts,
                        status="Absent",
                        notes="Auto-generated dummy record",
                        device_id=device_id,
                    )
                    db.session.add(rec)
                    db.session.flush()
                    created_att += 1

                    # Sometimes attach an excuse note (for admin features)
                    if rng.random() < 0.35:
                        note = ExcuseNote(
                            attendance_id=rec.id,
                            note="Seeded excuse note for demo purposes.",
                            created_by="system",
                            created_at=ts + timedelta(minutes=5),
                        )
                        db.session.add(note)
                        created_excuses += 1
                    continue

                if roll < base_absent + base_excused:
                    ts = _dt_at(day, sched_start) + timedelta(minutes=rng.randint(0, 20))
                    rec = Attendance(
                        user_id=u.id,
                        timestamp=ts,
                        status="Excused",
                        notes="Seeded excused record",
                        device_id=device_id,
                    )
                    db.session.add(rec)
                    created_att += 1
                    continue

                # Arrival
                is_late = rng.random() < base_late
                if is_late:
                    late_min = rng.randint(10, 120)
                    ts = _dt_at(day, sched_start) + timedelta(minutes=late_min)
                    status = rng.choice(["Late", "Tardy"])
                else:
                    ontime_min = rng.randint(-10, 40)
                    ts = _dt_at(day, sched_start) + timedelta(minutes=ontime_min)
                    status = rng.choice(["On Time", "Present"])

                rec = Attendance(
                    user_id=u.id,
                    timestamp=ts,
                    status=status,
                    notes=None,
                    device_id=device_id,
                )
                db.session.add(rec)
                db.session.flush()
                created_att += 1

                # Logout record for most arrivals
                if rng.random() < 0.92:
                    sched_end = u.schedule_end or "19:00"
                    logout_ts = _dt_at(day, sched_end) + timedelta(minutes=rng.randint(-15, 30))
                    db.session.add(
                        Attendance(
                            user_id=u.id,
                            timestamp=logout_ts,
                            status="Logout",
                            notes=None,
                            device_id=device_id,
                        )
                    )
                    created_att += 1

                # Occasionally create an edit record (e.g., admin corrected Late -> On Time)
                if rec.status in ("Late", "Tardy") and rng.random() < 0.08:
                    edit = AttendanceEdit(
                        attendance_id=rec.id,
                        previous_status=rec.status,
                        new_status="On Time",
                        previous_notes=rec.notes,
                        new_notes="Seeded correction for demo.",
                        edited_by="Admin Demo",
                        edited_at=rec.timestamp + timedelta(hours=2),
                    )
                    db.session.add(edit)
                    created_edits += 1
                    rec.status = "On Time"

        db.session.commit()

        print("Seed complete.")
        print(f"Users: {len(users)}")
        print(f"Attendance records created: {created_att}")
        print(f"Attendance edits created: {created_edits}")
        print(f"Excuse notes created: {created_excuses}")


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

