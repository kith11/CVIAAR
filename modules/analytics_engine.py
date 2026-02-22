
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .models import Attendance, User

class AnalyticsEngine:
    def __init__(self, db_session):
        self.db = db_session

    def get_attendance_dataframe(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Fetch attendance records with filters and convert to DataFrame."""
        query = self.db.query(Attendance, User).join(User, Attendance.user_id == User.id)
        
        if start_date:
            query = query.filter(Attendance.timestamp >= start_date)
        if end_date:
            # inclusive end date
            query = query.filter(Attendance.timestamp <= end_date + timedelta(days=1))
        if employment_type and employment_type != 'All':
            query = query.filter(User.employment_type == employment_type)
        if user_id and user_id != 'All':
            query = query.filter(Attendance.user_id == int(user_id))
            
        results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        data = []
        for log, user in results:
            data.append({
                'user_id': log.user_id,
                'name': user.name,
                'employment_type': user.employment_type,
                'timestamp': log.timestamp,
                'status': log.status,
                'hour': log.timestamp.hour,
                'weekday': log.timestamp.weekday()  # 0=Mon, 6=Sun
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = df['timestamp'].dt.date
        return df

    def get_weekly_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Calculate attendance counts by day of week."""
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return {'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                    'present': [0]*7, 'late': [0]*7, 'absent': [0]*7}

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Initialize counters
        present_counts = [0] * 7
        late_counts = [0] * 7
        absent_counts = [0] * 7

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

    def get_monthly_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Calculate attendance counts by day of month over the selected period."""
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return {'labels': [], 'present': [], 'late': [], 'absent': []}

        # If no date range provided, default to current month for labeling purposes
        if not start_date:
            now = datetime.now()
            import calendar
            num_days = calendar.monthrange(now.year, now.month)[1]
        else:
            # If range provided, we might want to just show 1-31 aggregations
            # or if it's a short range, show actual dates?
            # For simplicity, let's keep the "Day of Month" aggregation (1-31)
            # This allows seeing "Are people late more on the 1st vs 15th?" across multiple months if selected.
            num_days = 31 

        days = [str(i) for i in range(1, num_days + 1)]
        
        present_counts = [0] * num_days
        late_counts = [0] * num_days
        absent_counts = [0] * num_days

        for index, row in df.iterrows():
            day_idx = row['timestamp'].day - 1 # 0-indexed
            if day_idx >= num_days: continue # Should not happen with 31
            
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

    def get_peak_arrival_times(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Analyze when most people arrive."""
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return {}
            
        # Filter for arrival statuses
        arrivals = df[df['status'].isin(['Present', 'On Time', 'Late', 'Tardy'])]
        
        if arrivals.empty:
            return {}

        # Group by hour for simpler visualization
        # We can do 24-hour format
        hour_counts = arrivals['hour'].value_counts().sort_index()
        
        # Ensure all hours from min to max are represented
        if not hour_counts.empty:
            min_hour = int(hour_counts.index.min())
            max_hour = int(hour_counts.index.max())
            
            # Create dictionary with all hours in range
            result = {}
            for h in range(min_hour, max_hour + 1):
                result[h] = int(hour_counts.get(h, 0))
            return result
            
        return {}

    def get_status_distribution(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Get the distribution of attendance statuses."""
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return {
                'labels': [
                    'Late',
                    'Undertime',
                    'Official Time',
                    'Official Business',
                    'Leave',
                    'Leave Without Pay (LWOP)'
                ],
                'data': [0, 0, 0, 0, 0, 0],
                'colors': [
                    '#0d6efd',
                    '#6c757d',
                    '#adb5bd',
                    '#343a40',
                    '#dee2e6',
                    '#212529'
                ]
            }
            
        status_counts = df['status'].value_counts()

        categories = {
            'Late': 0,
            'Undertime': 0,
            'Official Time': 0,
            'Official Business': 0,
            'Leave': 0,
            'Leave Without Pay (LWOP)': 0
        }

        for status, count in status_counts.items():
            if status in ['Late', 'Tardy']:
                categories['Late'] += int(count)
            elif status in ['Present', 'On Time']:
                categories['Official Time'] += int(count)
            elif status == 'Absent':
                categories['Leave Without Pay (LWOP)'] += int(count)
            elif status == 'Excused':
                categories['Leave'] += int(count)

        labels = list(categories.keys())
        data = [categories[label] for label in labels]
        colors = [
            '#0d6efd',
            '#6c757d',
            '#adb5bd',
            '#343a40',
            '#dee2e6',
            '#212529'
        ]

        return {
            'labels': labels,
            'data': data,
            'colors': colors
        }
