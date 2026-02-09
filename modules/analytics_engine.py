
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .models import Attendance, User

class AnalyticsEngine:
    def __init__(self, db_session):
        self.db = db_session

    def get_attendance_dataframe(self):
        """Fetch all attendance records and convert to DataFrame."""
        logs = Attendance.query.all()
        if not logs:
            return pd.DataFrame()
        
        data = []
        for log in logs:
            data.append({
                'user_id': log.user_id,
                'timestamp': log.timestamp,
                'status': log.status,
                'hour': log.timestamp.hour,
                'weekday': log.timestamp.weekday()  # 0=Mon, 6=Sun
            })
        
        df = pd.DataFrame(data)
        df['date'] = df['timestamp'].dt.date
        return df

    def get_weekly_trends(self):
        """Calculate attendance counts by day of week."""
        df = self.get_attendance_dataframe()
        if df.empty:
            return {'labels': [], 'present': [], 'late': [], 'absent': []}

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Initialize counters
        present_counts = [0] * 7
        late_counts = [0] * 7
        absent_counts = [0] * 7

        # Group by weekday and status
        # Note: This is a simple count of events. 
        # For 'Absent', we rely on the database having explicit 'Absent' records.
        
        for index, row in df.iterrows():
            wd = row['weekday']
            status = row['status']
            
            if status in ['Present', 'On Time']:
                present_counts[wd] += 1
            elif status in ['Late', 'Tardy']:
                late_counts[wd] += 1
            elif status == 'Absent':
                absent_counts[wd] += 1

        return {
            'labels': days,
            'present': present_counts,
            'late': late_counts,
            'absent': absent_counts
        }

    def get_monthly_trends(self):
        """Calculate attendance counts by day of the current month."""
        df = self.get_attendance_dataframe()
        if df.empty:
            return {'labels': [], 'present': [], 'late': [], 'absent': []}

        # Filter for current month
        now = datetime.now()
        current_month_df = df[
            (df['timestamp'].dt.month == now.month) & 
            (df['timestamp'].dt.year == now.year)
        ]

        if current_month_df.empty:
            return {'labels': [], 'present': [], 'late': [], 'absent': []}

        # Get number of days in month
        import calendar
        num_days = calendar.monthrange(now.year, now.month)[1]
        days = [str(i) for i in range(1, num_days + 1)]
        
        present_counts = [0] * num_days
        late_counts = [0] * num_days
        absent_counts = [0] * num_days

        for index, row in current_month_df.iterrows():
            day_idx = row['timestamp'].day - 1 # 0-indexed
            status = row['status']
            
            if status in ['Present', 'On Time']:
                present_counts[day_idx] += 1
            elif status in ['Late', 'Tardy']:
                late_counts[day_idx] += 1
            elif status == 'Absent':
                absent_counts[day_idx] += 1

        return {
            'labels': days,
            'present': present_counts,
            'late': late_counts,
            'absent': absent_counts
        }

    def predict_risk_users(self):
        """Identify users at high risk of being late or absent."""
        df = self.get_attendance_dataframe()
        if df.empty:
            return []

        risk_report = []
        users = User.query.all()

        for user in users:
            user_logs = df[df['user_id'] == user.id]
            if user_logs.empty:
                continue

            total_days = len(user_logs)
            late_count = len(user_logs[user_logs['status'].isin(['Late', 'Tardy'])])
            absent_count = len(user_logs[user_logs['status'] == 'Absent'])
            
            late_rate = (late_count / total_days) * 100 if total_days > 0 else 0
            absent_rate = (absent_count / total_days) * 100 if total_days > 0 else 0

            # Simple Heuristic Prediction
            risk_level = "Low"
            prediction = "Likely On Time"
            
            if late_rate > 30:
                risk_level = "Medium"
                prediction = "Risk of Late Arrival"
            if absent_rate > 15 or late_rate > 50:
                risk_level = "High"
                prediction = "High Risk of Absence/Tardy"

            if risk_level != "Low":
                risk_report.append({
                    'name': user.name,
                    'late_rate': round(late_rate, 1),
                    'absent_rate': round(absent_rate, 1),
                    'risk_level': risk_level,
                    'prediction': prediction
                })

        # Sort by risk (High first)
        risk_map = {"High": 0, "Medium": 1, "Low": 2}
        risk_report.sort(key=lambda x: risk_map.get(x['risk_level'], 3))
        
        return risk_report

    def get_peak_arrival_times(self):
        """Analyze when most people arrive."""
        df = self.get_attendance_dataframe()
        if df.empty:
            return {'labels': [], 'data': []}
            
        # Filter for arrival statuses
        arrivals = df[df['status'].isin(['Present', 'On Time', 'Late', 'Tardy'])]
        
        if arrivals.empty:
            return {'labels': [], 'data': []}

        # Bin into 15-minute intervals
        # We'll focus on the morning window 6:00 - 9:00 for simplicity
        arrivals['minute_bin'] = arrivals['timestamp'].dt.hour * 60 + (arrivals['timestamp'].dt.minute // 15) * 15
        
        # Count occurrences
        counts = arrivals['minute_bin'].value_counts().sort_index()
        
        labels = []
        data = []
        
        for minutes, count in counts.items():
            h = minutes // 60
            m = minutes % 60
            time_str = f"{h:02d}:{m:02d}"
            labels.append(time_str)
            data.append(int(count))
            
        return {'labels': labels, 'data': data}
