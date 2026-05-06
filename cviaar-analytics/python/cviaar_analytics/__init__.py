"""
CVIAAR Analytics - High-performance analytics engine using Rust and DuckDB.

This module provides a Python interface to the high-performance Rust implementation
of CVIAAR's analytics functions, utilizing DuckDB for accelerated data processing.
"""

try:
    from .cviaar_analytics import (
        WeeklyTrends,
        MonthlyTrends,
        StatusDistribution,
        PeakArrivalTimes,
        RiskUser,
        get_weekly_trends,
        get_monthly_trends,
        get_status_distribution,
        get_peak_arrival_times,
        predict_risk_users
    )
    _HAS_RUST_BACKEND = True
except ImportError:
    _HAS_RUST_BACKEND = False

__all__ = [
    "WeeklyTrends",
    "MonthlyTrends", 
    "StatusDistribution",
    "PeakArrivalTimes",
    "RiskUser",
    "get_weekly_trends",
    "get_monthly_trends",
    "get_status_distribution",
    "get_peak_arrival_times",
    "predict_risk_users",
    "has_rust_backend"
]

def has_rust_backend():
    """Check if the Rust backend is available."""
    return _HAS_RUST_BACKEND