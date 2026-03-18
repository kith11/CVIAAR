use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct WeeklyTrends {
    #[pyo3(get)]
    pub labels: Vec<String>,
    #[pyo3(get)]
    pub present: Vec<i32>,
    #[pyo3(get)]
    pub late: Vec<i32>,
    #[pyo3(get)]
    pub absent: Vec<i32>,
}

#[pymethods]
impl WeeklyTrends {
    #[new]
    fn new(labels: Vec<String>, present: Vec<i32>, late: Vec<i32>, absent: Vec<i32>) -> Self {
        WeeklyTrends {
            labels,
            present,
            late,
            absent,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "WeeklyTrends(labels={:?}, present={:?}, late={:?}, absent={:?})",
            self.labels, self.present, self.late, self.absent
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct MonthlyTrends {
    #[pyo3(get)]
    pub labels: Vec<String>,
    #[pyo3(get)]
    pub present: Vec<i32>,
    #[pyo3(get)]
    pub late: Vec<i32>,
    #[pyo3(get)]
    pub absent: Vec<i32>,
}

#[pymethods]
impl MonthlyTrends {
    #[new]
    fn new(labels: Vec<String>, present: Vec<i32>, late: Vec<i32>, absent: Vec<i32>) -> Self {
        MonthlyTrends {
            labels,
            present,
            late,
            absent,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "MonthlyTrends(labels={:?}, present={:?}, late={:?}, absent={:?})",
            self.labels, self.present, self.late, self.absent
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct StatusDistribution {
    #[pyo3(get)]
    pub labels: Vec<String>,
    #[pyo3(get)]
    pub data: Vec<i32>,
}

#[pymethods]
impl StatusDistribution {
    #[new]
    fn new(labels: Vec<String>, data: Vec<i32>) -> Self {
        StatusDistribution { labels, data }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "StatusDistribution(labels={:?}, data={:?})",
            self.labels, self.data
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct PeakArrivalTimes {
    #[pyo3(get)]
    pub labels: Vec<String>,
    #[pyo3(get)]
    pub data: Vec<i32>,
}

#[pymethods]
impl PeakArrivalTimes {
    #[new]
    fn new(labels: Vec<String>, data: Vec<i32>) -> Self {
        PeakArrivalTimes { labels, data }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "PeakArrivalTimes(labels={:?}, data={:?})",
            self.labels, self.data
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct RiskUser {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub late_rate: String,
    #[pyo3(get)]
    pub absent_rate: String,
    #[pyo3(get)]
    pub risk_level: String,
    #[pyo3(get)]
    pub prediction: String,
}

#[pymethods]
impl RiskUser {
    #[new]
    fn new(
        name: String,
        late_rate: String,
        absent_rate: String,
        risk_level: String,
        prediction: String,
    ) -> Self {
        RiskUser {
            name,
            late_rate,
            absent_rate,
            risk_level,
            prediction,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "RiskUser(name={}, late_rate={}, absent_rate={}, risk_level={}, prediction={})",
            self.name, self.late_rate, self.absent_rate, self.risk_level, self.prediction
        )
    }
}