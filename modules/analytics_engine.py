
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .models import Attendance, User

class AnalyticsEngine:
    """
    A data processing engine for calculating attendance analytics.

    This class provides methods to fetch, filter, and analyze attendance data
    to generate insights such as weekly/monthly trends, user risk predictions,
    peak arrival times, and status distributions.
    """
    def __init__(self, db_session):
        """
        Initializes the AnalyticsEngine with a database session.

        Args:
            db_session: An active SQLAlchemy session.
        """
        self.db = db_session

    def get_attendance_dataframe(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Fetches attendance records from the database and converts them into a pandas DataFrame.

        This is a core utility method that allows for flexible filtering based on date range,
        employment type, and user ID. The resulting DataFrame includes user information and
        is enriched with temporal features like hour and weekday.

        Args:
            start_date (date, optional): The start date for the filter. Defaults to None.
            end_date (date, optional): The end date for the filter. Defaults to None.
            employment_type (str, optional): Filter by a specific employment type. Defaults to None.
            user_id (int, optional): Filter by a specific user ID. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered attendance records.
        """
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
            if not log.timestamp:
                continue
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
        """
        Calculates the total number of present, late, and absent records for each day of the week.

        This method is used to generate data for the weekly trend charts on the dashboard.

        Args:
            start_date (date, optional): The start date for the filter. Defaults to None.
            end_date (date, optional): The end date for the filter. Defaults to None.
            employment_type (str, optional): Filter by a specific employment type. Defaults to None.
            user_id (int, optional): Filter by a specific user ID. Defaults to None.

        Returns:
            dict: A dictionary containing labels for the days of the week and corresponding counts for each status.
        """
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return {'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                    'present': [0]*7, 'late': [0]*7, 'absent': [0]*7,
                    'comparison': {"growth": 0, "engagement_change": 0}}

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
            'absent': absent_counts,
            'comparison': self._get_weekly_comparison(start_date, end_date, employment_type, user_id)
        }

    def _get_weekly_comparison(self, start_date, end_date, employment_type, user_id):
        """Internal helper to calculate growth/engagement compared to the previous week."""
        if not start_date or not end_date:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
        
        # Current week metrics
        current_df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if current_df.empty:
            return {"growth": 0, "engagement_change": 0}
            
        current_present = len(current_df[current_df['status'].isin(['Present', 'On Time'])])
        current_total = len(current_df)
        current_rate = (current_present / current_total * 100) if current_total > 0 else 0

        # Previous week metrics
        prev_start = start_date - timedelta(days=7)
        prev_end = end_date - timedelta(days=7)
        prev_df = self.get_attendance_dataframe(prev_start, prev_end, employment_type, user_id)
        
        if prev_df.empty:
            return {"growth": 100, "engagement_change": current_rate}

        prev_present = len(prev_df[prev_df['status'].isin(['Present', 'On Time'])])
        prev_total = len(prev_df)
        prev_rate = (prev_present / prev_total * 100) if prev_total > 0 else 0

        growth = ((current_total - prev_total) / prev_total * 100) if prev_total > 0 else 0
        engagement_change = current_rate - prev_rate

        return {
            "growth": round(growth, 1),
            "engagement_change": round(engagement_change, 1),
            "current_total": current_total,
            "prev_total": prev_total
        }

    def get_monthly_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Calculates the total number of present, late, and absent records for each day of the month.

        This provides a day-by-day view of attendance patterns over a monthly period.

        Args:
            start_date (date, optional): The start date for the filter. Defaults to None.
            end_date (date, optional): The end date for the filter. Defaults to None.
            employment_type (str, optional): Filter by a specific employment type. Defaults to None.
            user_id (int, optional): Filter by a specific user ID. Defaults to None.

        Returns:
            dict: A dictionary containing labels for the days of the month and corresponding counts for each status.
        """
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return {'labels': [], 'present': [], 'late': [], 'absent': []}

        # If no date range provided, default to current month for labeling purposes
        if not start_date:
            # Default to Asia/Manila (PHT) for consistency
            import pytz
            now = datetime.now(pytz.timezone('Asia/Manila'))
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

    def predict_risk_users(self, days=90):
        """
        Identifies users at high risk of being late or absent based on their attendance history.
        By default, analyzes the last 90 days of data for better performance.

        This method uses a simple heuristic model:
        - **Medium Risk**: Late rate > 30%
        - **High Risk**: Absent rate > 15% or Late rate > 50%

        Returns:
            list: A list of dictionaries, where each dictionary represents a user at risk.
        """
        start_date = (datetime.now() - timedelta(days=days)).date()
        df = self.get_attendance_dataframe(start_date=start_date)
        if df.empty:
            return []

        risk_report = []
        users = self.db.query(User).all()

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
                    'late_rate': f"{round(late_rate, 1)}%",
                    'absent_rate': f"{round(absent_rate, 1)}%",
                    'risk_level': risk_level,
                    'prediction': prediction
                })

        # Sort by risk (High first)
        risk_map = {"High": 0, "Medium": 1, "Low": 2}
        risk_report.sort(key=lambda x: risk_map.get(x['risk_level'], 3))
        
        return risk_report

    def get_peak_arrival_times(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Analyzes attendance data to determine the peak arrival times.

        This method groups arrivals by the hour to identify when the most people are logging in.

        Args:
            start_date (date, optional): The start date for the filter. Defaults to None.
            end_date (date, optional): The end date for the filter. Defaults to None.
            employment_type (str, optional): Filter by a specific employment type. Defaults to None.
            user_id (int, optional): Filter by a specific user ID. Defaults to None.

        Returns:
            dict: A dictionary where keys are hours (0-23) and values are the count of arrivals.
        """
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
        """
        Calculates the distribution of different attendance statuses (e.g., Late, On Time, Absent).

        This is used to generate the pie chart on the analytics page.

        Args:
            start_date (date, optional): The start date for the filter. Defaults to None.
            end_date (date, optional): The end date for the filter. Defaults to None.
            employment_type (str, optional): Filter by a specific employment type. Defaults to None.
            user_id (int, optional): Filter by a specific user ID. Defaults to None.

        Returns:
            dict: A dictionary containing labels, data, and colors for the chart.
        """
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return {
                'labels': [
                    'Late',
                    'Undertime',
                    'Official Time',
                    'Official Business',
                    'Leave'
                ],
                'data': [0, 0, 0, 0, 0],
                'colors': [
                    '#0d6efd',
                    '#6c757d',
                    '#adb5bd',
                    '#343a40',
                    '#dee2e6'
                ]
            }
            
        status_counts = df['status'].value_counts()

        categories = {
            'Late': 0,
            'Undertime': 0,
            'Official Time': 0,
            'Official Business': 0,
            'Leave': 0
        }

        for status, count in status_counts.items():
            if status in ['Late', 'Tardy']:
                categories['Late'] += int(count)
            elif status in ['Present', 'On Time']:
                categories['Official Time'] += int(count)
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

    def get_advanced_insights(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Generates narrative insights based on attendance patterns, including anomalies and comparative indicators.
        """
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return ["No data available for the selected period."]

        insights = []
        
        # 1. Punctuality & Comparative Performance
        total = len(df)
        on_time = len(df[df['status'].isin(['Present', 'On Time'])])
        late = len(df[df['status'].isin(['Late', 'Tardy'])])
        punctuality_rate = (on_time / total * 100) if total > 0 else 0
        
        # Comparative indicator (vs 85% target)
        diff = punctuality_rate - 85
        comparison = "above" if diff >= 0 else "below"
        insights.append(f"Current punctuality rate is {punctuality_rate:.1f}%, which is {abs(diff):.1f}% {comparison} the organizational benchmark of 85%.")

        # 2. Peak Time & Anomaly Detection
        if not df.empty:
            arrivals = df[df['status'].isin(['Present', 'On Time', 'Late', 'Tardy'])]
            if not arrivals.empty:
                peak_hour = arrivals['hour'].mode().iloc[0]
                peak_count = len(arrivals[arrivals['hour'] == peak_hour])
                
                # Check for anomalies (e.g., unexpected late night check-ins)
                off_hours = arrivals[~arrivals['hour'].between(6, 20)]
                if not off_hours.empty:
                    insights.append(f"Anomaly detected: {len(off_hours)} check-ins occurred outside standard business hours (6AM-8PM).")
                
                insights.append(f"Peak arrival efficiency: {peak_count} check-ins consolidated at {peak_hour}:00, suggesting high terminal utilization during this window.")

        # 3. Weekly Trends & Engagement
        weekly = self.get_weekly_trends(start_date, end_date, employment_type, user_id)
        max_present = max(weekly['present'])
        if max_present > 0:
            best_day_idx = weekly['present'].index(max_present)
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Comparative weekly analysis
            avg_present = sum(weekly['present']) / 7
            performance = (max_present / avg_present * 100) - 100 if avg_present > 0 else 0
            insights.append(f"{days[best_day_idx]} is the highest engagement day, outperforming the weekly average by {performance:.1f}%.")

        return insights
