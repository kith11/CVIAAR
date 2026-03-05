#!/usr/bin/env python3
"""
Seed dummy staff and admin accounts. Run from repo root:
  python scripts/seed_dummy_users.py
"""
import os
import random
import sys

# Run from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from app import app, db
from modules.models import User


DUMMY_STAFF = [
    ("Maria Santos", "Full-time", "06:00", "19:00"),
    ("Juan Dela Cruz", "Full-time", "07:00", "18:00"),
    ("Ana Reyes", "Part-time", "08:00", "17:00"),
    ("Carlos Mendoza", "Full-time", "06:30", "18:30"),
    ("Elena Torres", "Full-time", "07:00", "16:00"),
    ("Miguel Fernandez", "Part-time", "09:00", "15:00"),
    ("Sofia Garcia", "Full-time", "06:00", "19:00"),
    ("Luis Ramirez", "Full-time", "07:30", "17:30"),
    ("Carmen Lopez", "Part-time", "08:00", "12:00"),
    ("Roberto Cruz", "Full-time", "06:00", "18:00"),
    ("Patricia Reyes", "Full-time", "07:00", "19:00"),
    ("Jose Martinez", "Part-time", "10:00", "14:00"),
    ("Rosa Hernandez", "Full-time", "06:30", "18:30"),
    ("Antonio Gonzales", "Full-time", "07:00", "17:00"),
    ("Teresa Diaz", "Part-time", "08:30", "12:30"),
]

DUMMY_ADMIN = ("Admin Demo", "Full-time", "06:00", "19:00")  # name, employment_type, start, end


def generate_staff_code():
    existing = {c for (c,) in db.session.query(User.staff_code).filter(User.staff_code.isnot(None)).all()}
    while True:
        code = f"{random.randint(0, 999999):06d}"
        if code not in existing:
            return code


def seed():
    with app.app_context():
        created = []
        faces_dir = os.path.join(REPO_ROOT, "data", "faces")
        os.makedirs(faces_dir, exist_ok=True)

        # One admin
        name, emp, start, end = DUMMY_ADMIN
        if User.query.filter(User.name == name).first() is None:
            admin = User(
                name=name,
                email="admin.demo@example.com",
                staff_code=generate_staff_code(),
                employment_type=emp,
                schedule_start=start,
                schedule_end=end,
                role="admin",
            )
            db.session.add(admin)
            db.session.flush()
            os.makedirs(os.path.join(faces_dir, str(admin.id)), exist_ok=True)
            created.append(f"  admin: {admin.name} (id={admin.id}, staff_code={admin.staff_code})")

        # Staff
        for name, emp, start, end in DUMMY_STAFF:
            if User.query.filter(User.name == name).first() is not None:
                continue
            code = generate_staff_code()
            user = User(
                name=name,
                email=f"{name.lower().replace(' ', '.')}@example.com",
                staff_code=code,
                employment_type=emp,
                schedule_start=start,
                schedule_end=end,
                role="staff",
            )
            db.session.add(user)
            db.session.flush()
            os.makedirs(os.path.join(faces_dir, str(user.id)), exist_ok=True)
            created.append(f"  staff: {user.name} (id={user.id}, staff_code={user.staff_code})")

        db.session.commit()
        if created:
            print("Created dummy accounts:")
            for line in created:
                print(line)
        else:
            print("No new accounts created (all dummies already exist).")


if __name__ == "__main__":
    seed()
