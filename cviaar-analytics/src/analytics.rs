use duckdb::{Connection, Result as DuckResult, params};
use crate::models::{WeeklyTrends, MonthlyTrends, StatusDistribution, PeakArrivalTimes, RiskUser};

pub struct AnalyticsEngine {
    conn: Connection,
}

impl AnalyticsEngine {
    pub fn new(db_path: &str) -> DuckResult<Self> {
        // Open an in-memory DuckDB connection
        let conn = Connection::open_in_memory()?;
        
        // Load the sqlite extension and attach the database
        // DuckDB's SQLite extension allows direct querying of SQLite files
        conn.execute("INSTALL sqlite; LOAD sqlite;", [])?;
        conn.execute(&format!("ATTACH '{}' AS sqlite_db (TYPE SQLITE);", db_path), [])?;
        
        Ok(AnalyticsEngine { conn })
    }

    pub fn get_weekly_trends(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> DuckResult<WeeklyTrends> {
        // Create a temporary view for our filtered data
        let mut where_clauses = vec![];
        
        if let Some(start) = start_date {
            where_clauses.push(format!("a.timestamp >= '{}'", start));
        }
        
        if let Some(end) = end_date {
            where_clauses.push(format!("a.timestamp <= '{}'", end));
        }
        
        if let Some(emp_type) = employment_type {
            if emp_type != "All" {
                where_clauses.push(format!("u.employment_type = '{}'", emp_type));
            }
        }
        
        if let Some(uid) = user_id {
            if uid != -1 { // -1 represents "All" in our system
                where_clauses.push(format!("a.user_id = {}", uid));
            }
        }
        
        let where_clause = if !where_clauses.is_empty() {
            format!("WHERE {}", where_clauses.join(" AND "))
        } else {
            String::new()
        };

        // Create the query for weekly trends using DuckDB's extract function
        let query = format!(
            "
            SELECT 
                CAST(extract('dow' FROM a.timestamp::TIMESTAMP) AS INTEGER) as weekday,
                SUM(CASE WHEN a.status IN ('Present', 'On Time') THEN 1 ELSE 0 END) as present_count,
                SUM(CASE WHEN a.status IN ('Late', 'Tardy') THEN 1 ELSE 0 END) as late_count,
                SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) as absent_count
            FROM sqlite_db.attendance_logs a
            JOIN sqlite_db.users u ON a.user_id = u.id
            {}
            GROUP BY weekday
            ORDER BY weekday
            ",
            where_clause
        );

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![], |row| {
            let weekday: i32 = row.get(0)?;
            let present: i32 = row.get(1)?;
            let late: i32 = row.get(2)?;
            let absent: i32 = row.get(3)?;
            Ok((weekday, present, late, absent))
        })?;

        // Initialize our result arrays
        let days = vec![
            "Sun".to_string(),
            "Mon".to_string(),
            "Tue".to_string(),
            "Wed".to_string(),
            "Thu".to_string(),
            "Fri".to_string(),
            "Sat".to_string(),
        ];
        
        let mut present_counts = vec![0; 7];
        let mut late_counts = vec![0; 7];
        let mut absent_counts = vec![0; 7];

        // Fill in the counts from our query results
        for row_result in rows {
            if let Ok((weekday, present, late, absent)) = row_result {
                if weekday >= 0 && weekday < 7 {
                    present_counts[weekday as usize] = present;
                    late_counts[weekday as usize] = late;
                    absent_counts[weekday as usize] = absent;
                }
            }
        }

        Ok(WeeklyTrends {
            labels: days,
            present: present_counts,
            late: late_counts,
            absent: absent_counts,
        })
    }

    pub fn get_monthly_trends(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> DuckResult<MonthlyTrends> {
        // Similar to weekly trends but grouped by day of month
        let mut where_clauses = vec![];
        
        if let Some(start) = start_date {
            where_clauses.push(format!("a.timestamp >= '{}'", start));
        }
        
        if let Some(end) = end_date {
            where_clauses.push(format!("a.timestamp <= '{}'", end));
        }
        
        if let Some(emp_type) = employment_type {
            if emp_type != "All" {
                where_clauses.push(format!("u.employment_type = '{}'", emp_type));
            }
        }
        
        if let Some(uid) = user_id {
            if uid != -1 {
                where_clauses.push(format!("a.user_id = {}", uid));
            }
        }
        
        let where_clause = if !where_clauses.is_empty() {
            format!("WHERE {}", where_clauses.join(" AND "))
        } else {
            String::new()
        };

        // For monthly trends, we'll use DuckDB's extract function for the day of month
        let query = format!(
            "
            SELECT 
                CAST(extract('day' FROM a.timestamp::TIMESTAMP) AS INTEGER) as day_of_month,
                SUM(CASE WHEN a.status IN ('Present', 'On Time') THEN 1 ELSE 0 END) as present_count,
                SUM(CASE WHEN a.status IN ('Late', 'Tardy') THEN 1 ELSE 0 END) as late_count,
                SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) as absent_count
            FROM sqlite_db.attendance_logs a
            JOIN sqlite_db.users u ON a.user_id = u.id
            {}
            GROUP BY day_of_month
            ORDER BY day_of_month
            ",
            where_clause
        );

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![], |row| {
            let day: i32 = row.get(0)?;
            let present: i32 = row.get(1)?;
            let late: i32 = row.get(2)?;
            let absent: i32 = row.get(3)?;
            Ok((day, present, late, absent))
        })?;

        // Initialize arrays for all days 1-31
        let days: Vec<String> = (1..=31).map(|d| d.to_string()).collect();
        let mut present_counts = vec![0; 31];
        let mut late_counts = vec![0; 31];
        let mut absent_counts = vec![0; 31];

        // Fill in the counts from our query results
        for row_result in rows {
            if let Ok((day, present, late, absent)) = row_result {
                if day >= 1 && day <= 31 {
                    let idx = (day - 1) as usize;
                    present_counts[idx] = present;
                    late_counts[idx] = late;
                    absent_counts[idx] = absent;
                }
            }
        }

        Ok(MonthlyTrends {
            labels: days,
            present: present_counts,
            late: late_counts,
            absent: absent_counts,
        })
    }

    pub fn get_status_distribution(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> DuckResult<StatusDistribution> {
        let mut where_clauses = vec![];
        
        if let Some(start) = start_date {
            where_clauses.push(format!("a.timestamp >= '{}'", start));
        }
        
        if let Some(end) = end_date {
            where_clauses.push(format!("a.timestamp <= '{}'", end));
        }
        
        if let Some(emp_type) = employment_type {
            if emp_type != "All" {
                where_clauses.push(format!("u.employment_type = '{}'", emp_type));
            }
        }
        
        if let Some(uid) = user_id {
            if uid != -1 {
                where_clauses.push(format!("a.user_id = {}", uid));
            }
        }
        
        let where_clause = if !where_clauses.is_empty() {
            format!("WHERE {}", where_clauses.join(" AND "))
        } else {
            String::new()
        };

        let query = format!(
            "
            SELECT 
                a.status,
                COUNT(*) as count
            FROM sqlite_db.attendance_logs a
            JOIN sqlite_db.users u ON a.user_id = u.id
            {}
            GROUP BY a.status
            ORDER BY count DESC
            ",
            where_clause
        );

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![], |row| {
            let status: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((status, count as i32))
        })?;

        let mut labels = Vec::new();
        let mut data = Vec::new();

        for row_result in rows {
            if let Ok((status, count)) = row_result {
                labels.push(status);
                data.push(count);
            }
        }

        // Map specific statuses to broader categories as in the Python version
        let mut categorized_labels = vec![
            "Late".to_string(),
            "Undertime".to_string(),
            "Official Time".to_string(),
            "Official Business".to_string(),
            "Leave".to_string(),
        ];
        
        let mut categorized_data = vec![0; 5];

        for (i, status) in labels.iter().enumerate() {
            let count = data[i];
            match status.as_str() {
                "Late" | "Tardy" => categorized_data[0] += count,
                "Present" | "On Time" => categorized_data[2] += count,
                "Excused" => categorized_data[4] += count,
                _ => {
                    // For other statuses, we'll map them to appropriate categories
                    // or add them as separate entries
                    categorized_labels.push(status.clone());
                    categorized_data.push(count);
                }
            }
        }

        Ok(StatusDistribution {
            labels: categorized_labels,
            data: categorized_data,
        })
    }

    pub fn get_peak_arrival_times(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        employment_type: Option<&str>,
        user_id: Option<i32>,
    ) -> DuckResult<PeakArrivalTimes> {
        let mut where_clauses = vec![];
        
        if let Some(start) = start_date {
            where_clauses.push(format!("a.timestamp >= '{}'", start));
        }
        
        if let Some(end) = end_date {
            where_clauses.push(format!("a.timestamp <= '{}'", end));
        }
        
        if let Some(emp_type) = employment_type {
            if emp_type != "All" {
                where_clauses.push(format!("u.employment_type = '{}'", emp_type));
            }
        }
        
        if let Some(uid) = user_id {
            if uid != -1 {
                where_clauses.push(format!("a.user_id = {}", uid));
            }
        }
        
        let where_clause_str = if !where_clauses.is_empty() {
            format!("WHERE {}", where_clauses.join(" AND "))
        } else {
            String::new()
        };

        // For DuckDB query, we need to handle the WHERE clause correctly
        let and_clause = if where_clause_str.is_empty() {
            String::new()
        } else {
            format!("AND {}", &where_clause_str[6..])
        };

        // Filter for arrival statuses and group by hour using DuckDB's extract function
        let query = format!(
            "
            SELECT 
                CAST(extract('hour' FROM a.timestamp::TIMESTAMP) AS INTEGER) as hour,
                COUNT(*) as count
            FROM sqlite_db.attendance_logs a
            JOIN sqlite_db.users u ON a.user_id = u.id
            WHERE a.status IN ('Present', 'On Time', 'Late', 'Tardy')
            {}
            GROUP BY hour
            ORDER BY hour
            ",
            and_clause
        );

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map(params![], |row| {
            let hour: i32 = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((hour, count as i32))
        })?;

        // Initialize arrays for all hours 0-23
        let labels: Vec<String> = (0..24).map(|h| format!("{}:00", h)).collect();
        let mut data = vec![0; 24];

        // Fill in the counts from our query results
        for row_result in rows {
            if let Ok((hour, count)) = row_result {
                if hour >= 0 && hour < 24 {
                    data[hour as usize] = count;
                }
            }
        }

        Ok(PeakArrivalTimes {
            labels,
            data,
        })
    }

    pub fn predict_risk_users(&self) -> DuckResult<Vec<RiskUser>> {
        // This is a simplified version of the risk prediction algorithm
        // In a full implementation, we would implement the same logic as in Python
        let query = "
            SELECT 
                u.name,
                u.id,
                COUNT(a.id) as total_records,
                SUM(CASE WHEN a.status IN ('Late', 'Tardy') THEN 1 ELSE 0 END) as late_count,
                SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) as absent_count
            FROM sqlite_db.users u
            LEFT JOIN sqlite_db.attendance_logs a ON u.id = a.user_id
            GROUP BY u.id, u.name
            HAVING total_records > 0
        ";

        let mut stmt = self.conn.prepare(query)?;
        let rows = stmt.query_map(params![], |row| {
            let name: String = row.get(0)?;
            let _user_id: i32 = row.get(1)?;
            let total_records: i64 = row.get(2)?;
            let late_count: i64 = row.get(3)?;
            let absent_count: i64 = row.get(4)?;
            Ok((name, total_records as f64, late_count as f64, absent_count as f64))
        })?;

        let mut risk_users = Vec::new();

        for row_result in rows {
            if let Ok((name, total, late, absent)) = row_result {
                let late_rate = (late / total) * 100.0;
                let absent_rate = (absent / total) * 100.0;
                
                let (risk_level, prediction) = if absent_rate > 20.0 || late_rate > 50.0 {
                    ("High".to_string(), "Likely to continue poor attendance".to_string())
                } else if absent_rate > 10.0 || late_rate > 25.0 {
                    ("Medium".to_string(), "Monitor attendance closely".to_string())
                } else {
                    ("Low".to_string(), "Stable attendance pattern".to_string())
                };

                risk_users.push(RiskUser {
                    name,
                    late_rate: format!("{:.1}%", late_rate),
                    absent_rate: format!("{:.1}%", absent_rate),
                    risk_level,
                    prediction,
                });
            }
        }

        Ok(risk_users)
    }
}