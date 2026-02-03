from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    schedule_start = db.Column(db.String(5), default="06:00")
    schedule_end = db.Column(db.String(5), default="19:00")
    attendances = db.relationship('Attendance', backref='user', lazy=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    status = db.Column(db.String(20), nullable=False) # On Time, Late, Absent, Excused, Logout
    notes = db.Column(db.String(200), nullable=True)
