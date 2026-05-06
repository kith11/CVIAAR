"""
CVIAAR Analytics Engine - Abstraction layer for analytics processing.

This module provides an abstraction layer that can use either the Python implementation
or the high-performance Rust implementation with DuckDB, depending on availability
and configuration.
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime

# Try to import the Rust backend
try:
    from .cviaar_analytics import (
        get_weekly_trends as rust_get_weekly_trends,
        get_monthly_trends as rust_get_monthly_trends,
        get_status_distribution as rust_get_status_distribution,
        get_peak_arrival_times as rust_get_peak_arrival_times,
        predict_risk_users as rust_predict_risk_users
    )
    HAS_RUST_BACKEND = True
except ImportError:
    HAS_RUST_BACKEND = False

class AnalyticsEngine:
    """
    Analytics engine that can use either Python or Rust backend.
    
    This class provides a unified interface for analytics operations that can
    switch between the original Python implementation and the high-performance
    Rust implementation with DuckDB.
    """
    
    def __init__(self, db_session, use_rust_backend: bool = True, db_path: str = None):
        """
        Initialize the analytics engine.
        
        Args:
            db_session: SQLAlchemy database session
            use_rust_backend: Whether to use the Rust backend if available
            db_path: Path to SQLite database file for Rust backend
        """
        self.db = db_session
        self.use_rust_backend = use_rust_backend and HAS_RUST_BACKEND
        self.db_path = db_path or self._get_db_path()
        
        # Import the original Python implementation if needed
        if not self.use_rust_backend:
            # We'll import this dynamically when needed
            pass
    
    def _get_db_path(self) -> str:
        """Get the path to the SQLite database file."""
        # This should match the path used in the CVIAAR application
        import os
        basedir = os.path.abspath(os.path.dirname(__file__))
        # Adjust this path to match your actual database location
        return os.path.join(basedir, "..", "..", "data", "offline", "cviaar_local.sqlite3")
    
    def get_weekly_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Calculate weekly trends for attendance data.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            employment_type: Employment type filter (optional)
            user_id: User ID filter (optional)
            
        Returns:
            WeeklyTrends object with labels and counts
        """
        if self.use_rust_backend and HAS_RUST_BACKEND:
            # Convert parameters to the format expected by Rust
            start_str = start_date.strftime('%Y-%m-%d') if start_date else None
            end_str = end_date.strftime('%Y-%m-%d') if end_date else None
            user_id_int = int(user_id) if user_id and user_id != "All" else None
            
            try:
                return rust_get_weekly_trends(
                    self.db_path, start_str, end_str, employment_type, user_id_int
                )
            except Exception as e:
                # Fall back to Python implementation if Rust fails
                print(f"Rust backend failed, falling back to Python: {e}")
                self.use_rust_backend = False
        
        # Fallback to Python implementation
        return self._python_get_weekly_trends(start_date, end_date, employment_type, user_id)
    
    def get_monthly_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Calculate monthly trends for attendance data.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            employment_type: Employment type filter (optional)
            user_id: User ID filter (optional)
            
        Returns:
            MonthlyTrends object with labels and counts
        """
        if self.use_rust_backend and HAS_RUST_BACKEND:
            # Convert parameters to the format expected by Rust
            start_str = start_date.strftime('%Y-%m-%d') if start_date else None
            end_str = end_date.strftime('%Y-%m-%d') if end_date else None
            user_id_int = int(user_id) if user_id and user_id != "All" else None
            
            try:
                return rust_get_monthly_trends(
                    self.db_path, start_str, end_str, employment_type, user_id_int
                )
            except Exception as e:
                # Fall back to Python implementation if Rust fails
                print(f"Rust backend failed, falling back to Python: {e}")
                self.use_rust_backend = False
        
        # Fallback to Python implementation
        return self._python_get_monthly_trends(start_date, end_date, employment_type, user_id)
    
    def get_status_distribution(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Calculate status distribution for attendance data.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            employment_type: Employment type filter (optional)
            user_id: User ID filter (optional)
            
        Returns:
            StatusDistribution object with labels and data
        """
        if self.use_rust_backend and HAS_RUST_BACKEND:
            # Convert parameters to the format expected by Rust
            start_str = start_date.strftime('%Y-%m-%d') if start_date else None
            end_str = end_date.strftime('%Y-%m-%d') if end_date else None
            user_id_int = int(user_id) if user_id and user_id != "All" else None
            
            try:
                return rust_get_status_distribution(
                    self.db_path, start_str, end_str, employment_type, user_id_int
                )
            except Exception as e:
                # Fall back to Python implementation if Rust fails
                print(f"Rust backend failed, falling back to Python: {e}")
                self.use_rust_backend = False
        
        # Fallback to Python implementation
        return self._python_get_status_distribution(start_date, end_date, employment_type, user_id)
    
    def get_peak_arrival_times(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """
        Calculate peak arrival times for attendance data.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            employment_type: Employment type filter (optional)
            user_id: User ID filter (optional)
            
        Returns:
            PeakArrivalTimes object with labels and data
        """
        if self.use_rust_backend and HAS_RUST_BACKEND:
            # Convert parameters to the format expected by Rust
            start_str = start_date.strftime('%Y-%m-%d') if start_date else None
            end_str = end_date.strftime('%Y-%m-%d') if end_date else None
            user_id_int = int(user_id) if user_id and user_id != "All" else None
            
            try:
                return rust_get_peak_arrival_times(
                    self.db_path, start_str, end_str, employment_type, user_id_int
                )
            except Exception as e:
                # Fall back to Python implementation if Rust fails
                print(f"Rust backend failed, falling back to Python: {e}")
                self.use_rust_backend = False
        
        # Fallback to Python implementation
        return self._python_get_peak_arrival_times(start_date, end_date, employment_type, user_id)
    
    def predict_risk_users(self):
        """
        Predict risk users based on attendance patterns.
        
        Returns:
            List of RiskUser objects
        """
        if self.use_rust_backend and HAS_RUST_BACKEND:
            try:
                return rust_predict_risk_users(self.db_path)
            except Exception as e:
                # Fall back to Python implementation if Rust fails
                print(f"Rust backend failed, falling back to Python: {e}")
                self.use_rust_backend = False
        
        # Fallback to Python implementation
        return self._python_predict_risk_users()
    
    # Placeholder methods for Python implementations
    # In a real implementation, these would contain the original Python code
    
    def _python_get_weekly_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Fallback to original Python implementation."""
        # This would contain the original implementation from modules/analytics_engine.py
        # For now, we'll just raise NotImplementedError to indicate this needs to be implemented
        raise NotImplementedError("Python implementation not included in this proof of concept")
    
    def _python_get_monthly_trends(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Fallback to original Python implementation."""
        raise NotImplementedError("Python implementation not included in this proof of concept")
    
    def _python_get_status_distribution(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Fallback to original Python implementation."""
        raise NotImplementedError("Python implementation not included in this proof of concept")
    
    def _python_get_peak_arrival_times(self, start_date=None, end_date=None, employment_type=None, user_id=None):
        """Fallback to original Python implementation."""
        raise NotImplementedError("Python implementation not included in this proof of concept")
    
    def _python_predict_risk_users(self):
        """Fallback to original Python implementation."""
        raise NotImplementedError("Python implementation not included in this proof of concept")