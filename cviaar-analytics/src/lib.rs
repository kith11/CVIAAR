use pyo3::prelude::*;

mod models;
mod analytics;

use models::{WeeklyTrends, MonthlyTrends, StatusDistribution, PeakArrivalTimes, RiskUser};
use analytics::AnalyticsEngine;

/// A Python module implemented in Rust for CVIAAR analytics.
#[pymodule]
fn cviaar_analytics(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WeeklyTrends>()?;
    m.add_class::<MonthlyTrends>()?;
    m.add_class::<StatusDistribution>()?;
    m.add_class::<PeakArrivalTimes>()?;
    m.add_class::<RiskUser>()?;
    
    /// Get weekly trends analytics data
    #[pyfunction]
    fn get_weekly_trends(
        db_path: &str,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> PyResult<WeeklyTrends> {
        let engine = AnalyticsEngine::new(db_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        let result = engine.get_weekly_trends(start_date, end_date, employment_type, user_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        Ok(result)
    }
    
    /// Get monthly trends analytics data
    #[pyfunction]
    fn get_monthly_trends(
        db_path: &str,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> PyResult<MonthlyTrends> {
        let engine = AnalyticsEngine::new(db_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        let result = engine.get_monthly_trends(start_date, end_date, employment_type, user_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        Ok(result)
    }
    
    /// Get status distribution analytics data
    #[pyfunction]
    fn get_status_distribution(
        db_path: &str,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> PyResult<StatusDistribution> {
        let engine = AnalyticsEngine::new(db_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        let result = engine.get_status_distribution(start_date, end_date, employment_type, user_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        Ok(result)
    }
    
    /// Get peak arrival times analytics data
    #[pyfunction]
    fn get_peak_arrival_times(
        db_path: &str,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> PyResult<PeakArrivalTimes> {
        let engine = AnalyticsEngine::new(db_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        let result = engine.get_peak_arrival_times(start_date, end_date, employment_type, user_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        Ok(result)
    }
    
    /// Get risk prediction analytics data
    #[pyfunction]
    fn predict_risk_users(
        db_path: &str,
    ) -> PyResult<Vec<RiskUser>> {
        let engine = AnalyticsEngine::new(db_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        let result = engine.predict_risk_users()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        
        Ok(result)
    }
    
    // Add functions to module
    m.add_function(wrap_pyfunction!(get_weekly_trends, m)?)?;
    m.add_function(wrap_pyfunction!(get_monthly_trends, m)?)?;
    m.add_function(wrap_pyfunction!(get_status_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(get_peak_arrival_times, m)?)?;
    m.add_function(wrap_pyfunction!(predict_risk_users, m)?)?;
    
    Ok(())
}