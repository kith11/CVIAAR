
import calendar
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .attendance_rules import (
    ABSENT_STATUSES,
    is_absent_status,
    is_late_status,
    is_login_record,
    is_logout_record,
    is_on_time_status,
    is_present_status,
)
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
                'id': log.id,
                'user_id': log.user_id,
                'name': user.name,
                'employment_type': user.employment_type,
                'schedule_start': user.schedule_start or '06:00',
                'schedule_end': user.schedule_end or '19:00',
                'timestamp': log.timestamp,
                'status': log.status,
                'session': getattr(log, 'session', None),
                'event_type': getattr(log, 'event_type', None),
                'auto_generated': getattr(log, 'auto_generated', 0),
                'device_id': getattr(log, 'device_id', None),
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
            
            if is_on_time_status(status):
                present_counts[wd] += 1
            elif is_late_status(status):
                late_counts[wd] += 1
            elif is_absent_status(status):
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
            
        current_present = len(current_df[current_df['status'].apply(is_on_time_status)])
        current_total = len(current_df)
        current_rate = (current_present / current_total * 100) if current_total > 0 else 0

        # Previous week metrics
        prev_start = start_date - timedelta(days=7)
        prev_end = end_date - timedelta(days=7)
        prev_df = self.get_attendance_dataframe(prev_start, prev_end, employment_type, user_id)
        
        if prev_df.empty:
            return {"growth": 100, "engagement_change": current_rate}

        prev_present = len(prev_df[prev_df['status'].apply(is_on_time_status)])
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
            
            if is_on_time_status(status):
                present_counts[day_idx] += 1
            elif is_late_status(status):
                late_counts[day_idx] += 1
            elif is_absent_status(status):
                absent_counts[day_idx] += 1
                
        return {
            'labels': days,
            'present': present_counts,
            'late': late_counts,
            'absent': absent_counts
        }

    def predict_risk_users(self, start_date=None, end_date=None, employment_type=None, user_id=None, days=90):
        """
        Identifies staff at elevated risk of lateness or absence within the filtered period.

        - **Medium Risk**: Late rate > 30%
        - **High Risk**: Absent rate > 15% or Late rate > 50%
        """
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        if df.empty:
            return []

        risk_report = []
        user_query = self.db.query(User)
        if employment_type and employment_type != 'All':
            user_query = user_query.filter(User.employment_type == employment_type)
        if user_id and user_id != 'All':
            user_query = user_query.filter(User.id == int(user_id))
        users = user_query.all()

        for user in users:
            user_logs = df[df['user_id'] == user.id]
            if user_logs.empty:
                continue

            total_days = len(user_logs)
            late_count = len(user_logs[user_logs['status'].apply(is_late_status)])
            absent_count = len(user_logs[user_logs['status'].isin(list(ABSENT_STATUSES))])
            
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
                    'user_id': user.id,
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
                    'Absent',
                    'Undertime',
                    'On Time',
                    'Official Business',
                    'Excused Absence',
                ],
                'data': [0, 0, 0, 0, 0, 0],
                'colors': [
                    '#e67e22',
                    '#e74c3c',
                    '#6c757d',
                    '#1e90ff',
                    '#343a40',
                    '#9b59b6',
                ]
            }
            
        status_counts = df['status'].value_counts()

        categories = {
            'Late': 0,
            'Absent': 0,
            'Undertime': 0,
            'On Time': 0,
            'Official Business': 0,
            'Excused Absence': 0,
        }

        for status, count in status_counts.items():
            if is_late_status(status):
                categories['Late'] += int(count)
            elif is_absent_status(status):
                categories['Absent'] += int(count)
            elif is_on_time_status(status):
                categories['On Time'] += int(count)
            elif status == 'Excused':
                categories['Excused Absence'] += int(count)

        labels = list(categories.keys())
        data = [categories[label] for label in labels]
        colors = [
            '#0d6efd',
            '#dc3545',
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
            return ["No attendance records were found for the selected dates."]

        insights = []
        
        # 1. Punctuality & Comparative Performance
        total = len(df)
        on_time = len(df[df['status'].apply(is_on_time_status)])
        late = len(df[df['status'].apply(is_late_status)])
        punctuality_rate = (on_time / total * 100) if total > 0 else 0
        
        # Comparative indicator (vs 85% target)
        diff = punctuality_rate - 85
        if diff >= 0:
            insights.append(
                f"{punctuality_rate:.1f}% of recorded check-ins were on time. "
                f"That is {abs(diff):.1f}% above the 85% punctuality goal."
            )
        else:
            insights.append(
                f"{punctuality_rate:.1f}% of recorded check-ins were on time. "
                f"That is {abs(diff):.1f}% below the 85% punctuality goal."
            )

        # 2. Peak Time & Anomaly Detection
        if not df.empty:
            arrivals = df[df['status'].isin(['Present', 'On Time', 'Late', 'Tardy'])]
            if not arrivals.empty:
                peak_hour = arrivals['hour'].mode().iloc[0]
                peak_count = len(arrivals[arrivals['hour'] == peak_hour])
                
                # Check for anomalies (e.g., unexpected late night check-ins)
                off_hours = arrivals[~arrivals['hour'].between(6, 20)]
                if not off_hours.empty:
                    insights.append(
                        f"{len(off_hours)} check-ins happened outside the usual 6:00 AM to 8:00 PM window. "
                        "These may be worth reviewing."
                    )
                
                peak_label = f"{peak_hour:02d}:00"
                insights.append(
                    f"The busiest check-in time was around {peak_label}, with {peak_count} attendance logs recorded at that hour."
                )

        # 3. Weekly Trends & Engagement
        weekly = self.get_weekly_trends(start_date, end_date, employment_type, user_id)
        max_present = max(weekly['present'])
        if max_present > 0:
            best_day_idx = weekly['present'].index(max_present)
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Comparative weekly analysis
            avg_present = sum(weekly['present']) / 7
            performance = (max_present / avg_present * 100) - 100 if avg_present > 0 else 0
            insights.append(
                f"{days[best_day_idx]} had the strongest attendance in this view, "
                f"at {performance:.1f}% above the weekly average."
            )

        return insights

    @staticmethod
    def _parse_hhmm(value: str) -> tuple[int, int]:
        try:
            hour, minute = value.split(':')
            return int(hour), int(minute)
        except (ValueError, AttributeError):
            return 6, 0

    def get_kpi_summary(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Top-row KPI cards for school staff attendance."""
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        empty = {
            'login_late': 0,
            'login_early': 0,
            'overtime_hours': 0,
            'absent_sessions': 0,
            'attendance_percentage': 0.0,
            'avg_daily_hours': 0.0,
        }
        if df.empty:
            return empty

        login_df = df[df.apply(
            lambda row: is_login_record(row['status'], row.get('event_type')),
            axis=1
        )]

        late_count = int(login_df['status'].apply(is_late_status).sum())
        early_count = 0
        for _, row in login_df.iterrows():
            sched_h, sched_m = self._parse_hhmm(row['schedule_start'])
            sched_minutes = sched_h * 60 + sched_m
            actual_minutes = row['timestamp'].hour * 60 + row['timestamp'].minute
            if actual_minutes < sched_minutes:
                early_count += 1

        overtime_minutes = 0
        logout_df = df[df.apply(
            lambda row: is_logout_record(row['status'], row.get('event_type')),
            axis=1
        )]
        for _, row in logout_df.iterrows():
            end_h, end_m = self._parse_hhmm(row['schedule_end'])
            end_minutes = end_h * 60 + end_m
            actual_minutes = row['timestamp'].hour * 60 + row['timestamp'].minute
            if actual_minutes > end_minutes:
                overtime_minutes += actual_minutes - end_minutes

        absent_sessions = int(df['status'].apply(is_absent_status).sum())

        on_time = int(df['status'].apply(is_on_time_status).sum())
        late = int(df['status'].apply(is_late_status).sum())
        absent = int(df['status'].apply(is_absent_status).sum())
        attendance_total = on_time + late + absent
        attendance_pct = round((on_time / attendance_total * 100), 1) if attendance_total > 0 else 0.0

        worked_hours = []
        for user_id_val, user_df in df.groupby('user_id'):
            for day, day_df in user_df.groupby('date'):
                logins = day_df[day_df.apply(
                    lambda row: is_login_record(row['status'], row.get('event_type')),
                    axis=1
                )].sort_values('timestamp')
                logouts = day_df[day_df.apply(
                    lambda row: is_logout_record(row['status'], row.get('event_type')),
                    axis=1
                )].sort_values('timestamp')
                if logins.empty or logouts.empty:
                    continue
                delta = (logouts.iloc[-1]['timestamp'] - logins.iloc[0]['timestamp']).total_seconds() / 3600
                if delta > 0:
                    worked_hours.append(delta)

        avg_worked = round(sum(worked_hours) / len(worked_hours), 1) if worked_hours else 0.0

        return {
            'login_late': late_count,
            'login_early': early_count,
            'overtime_hours': round(overtime_minutes / 60, 1),
            'absent_sessions': absent_sessions,
            'attendance_percentage': attendance_pct,
            'avg_daily_hours': avg_worked,
        }

    def get_working_location(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """On-campus kiosk vs other device check-ins."""
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        login_df = df[df.apply(
            lambda row: is_login_record(row['status'], row.get('event_type')),
            axis=1
        )] if not df.empty else df

        if login_df.empty:
            return {'labels': ['On Campus', 'Other Device'], 'values': [0, 0]}

        device_counts = login_df['device_id'].fillna('unknown').value_counts()
        if len(device_counts) <= 1:
            on_campus = int(len(login_df))
            other = 0
        else:
            primary_device = device_counts.index[0]
            on_campus = int((login_df['device_id'].fillna('unknown') == primary_device).sum())
            other = int(len(login_df) - on_campus)

        return {'labels': ['On Campus', 'Other Device'], 'values': [on_campus, other]}

    def _empty_month_stats(self):
        return {
            'available': 0,
            'overtime': 0,
            'early_clock_in': 0,
            'late_clock_in': 0,
            'absent': 0,
            'early_clock_out': 0,
        }

    def _compute_month_bucket(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {**self._empty_month_stats(), 'overall': 0.0, 'absenteeism': 0.0}

        on_time = int(df['status'].apply(is_on_time_status).sum())
        late = int(df['status'].apply(is_late_status).sum())
        absent = int(df['status'].apply(is_absent_status).sum())
        total = on_time + late + absent
        overall = round((on_time / total * 100), 1) if total > 0 else 0.0
        absenteeism = round((absent / len(df) * 100), 2) if len(df) > 0 else 0.0

        early_in = 0
        login_df = df[df.apply(
            lambda row: is_login_record(row['status'], row.get('event_type')),
            axis=1
        )]
        for _, row in login_df.iterrows():
            sched_h, sched_m = self._parse_hhmm(row['schedule_start'])
            actual = row['timestamp'].hour * 60 + row['timestamp'].minute
            if actual < sched_h * 60 + sched_m:
                early_in += 1

        early_out = 0
        overtime = 0
        logout_df = df[df.apply(
            lambda row: is_logout_record(row['status'], row.get('event_type')),
            axis=1
        )]
        for _, row in logout_df.iterrows():
            end_h, end_m = self._parse_hhmm(row['schedule_end'])
            actual = row['timestamp'].hour * 60 + row['timestamp'].minute
            if actual < end_h * 60 + end_m:
                early_out += 1
            if actual > end_h * 60 + end_m:
                overtime += 1

        return {
            'overall': overall,
            'absenteeism': absenteeism,
            'available': on_time,
            'overtime': overtime,
            'early_clock_in': early_in,
            'late_clock_in': late,
            'absent': absent,
            'early_clock_out': early_out,
        }

    def get_six_month_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Six-month attendance trends (single DB query, anchored to filter end date)."""
        anchor = end_date or datetime.now().date()
        months = []
        labels = []
        for offset in range(5, -1, -1):
            month_anchor = datetime(anchor.year, anchor.month, 1) - pd.DateOffset(months=offset)
            months.append((int(month_anchor.year), int(month_anchor.month)))
            labels.append(month_anchor.strftime('%b'))

        span_start = datetime(months[0][0], months[0][1], 1).date()
        y, m = months[-1]
        span_end = datetime(y, m, calendar.monthrange(y, m)[1]).date()
        df = self.get_attendance_dataframe(span_start, span_end, employment_type, user_id)

        period_label = f"{labels[0]} – {labels[-1]} {months[-1][0]}"

        stat_keys = ['available', 'overtime', 'early_clock_in', 'late_clock_in', 'absent', 'early_clock_out']
        stats = {key: [] for key in stat_keys}
        overall = []
        absenteeism = []

        if df.empty:
            for _ in months:
                overall.append(0.0)
                absenteeism.append(0.0)
                for key in stat_keys:
                    stats[key].append(0)
            return {
                'labels': labels,
                'period_label': period_label,
                'overall_attendance': overall,
                'absenteeism_rate': absenteeism,
                'attendance_statistics': stats,
            }

        df = df.copy()
        df['year_month'] = list(zip(df['timestamp'].dt.year, df['timestamp'].dt.month))

        for year, month in months:
            bucket_df = df[df['year_month'] == (year, month)]
            metrics = self._compute_month_bucket(bucket_df)
            overall.append(metrics['overall'])
            absenteeism.append(metrics['absenteeism'])
            for key in stat_keys:
                stats[key].append(metrics[key])

        return {
            'labels': labels,
            'period_label': period_label,
            'overall_attendance': overall,
            'absenteeism_rate': absenteeism,
            'attendance_statistics': stats,
        }

    def get_attendance_heatmap(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Day-of-week x hour density matrix for check-in activity."""
        df = self.get_attendance_dataframe(start_date, end_date, employment_type, user_id)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hours = [f'{h:02d}' for h in range(24)]

        matrix = [[0 for _ in range(24)] for _ in range(7)]
        if df.empty:
            return {'days': days, 'hours': hours, 'values': matrix, 'max': 0}

        arrivals = df[df.apply(
            lambda row: is_login_record(row['status'], row.get('event_type')),
            axis=1
        )]
        for _, row in arrivals.iterrows():
            matrix[row['weekday']][row['hour']] += 1

        max_val = max(max(row) for row in matrix) if arrivals is not None and not arrivals.empty else 0
        return {'days': days, 'hours': hours, 'values': matrix, 'max': max_val}
