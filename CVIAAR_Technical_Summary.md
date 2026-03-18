# CVIAAR AI Biometric Attendance System - Technical Summary

## 1. Project Overview
CVIAAR is an advanced, AI-powered biometric attendance management system designed for high-accuracy identity verification and real-time attendance tracking. It leverages computer vision and machine learning to provide a seamless, secure, and automated experience for both administrators and staff members.

## 2. Core Features

### Biometric Authentication & Kiosk
- **Facial Recognition**: High-performance recognition using LBPH (Local Binary Patterns Histograms) with adaptive thresholding.
- **Liveness Detection**: Real-time blink detection (Eye Aspect Ratio - EAR) to prevent spoofing and ensure physical presence.
- **Standby Mode**: Intelligent camera state management that wakes up for authentication and returns to a low-resource standby state.
- **Automated Enrollment**: Multi-angle image capture for training new user profiles with boundary validation.

### Analytics & Reporting
- **Admin Dashboard**: Modern, compact UI with real-time status broadcasting via Redis.
- **Interactive Visualizations**: Dynamic D3.js and Plotly charts showing weekly attendance trends, punctuality scores, and peak arrival windows.
- **Advanced Engine**: High-performance analytics engine (Rust-powered backend) for fast data processing.
- **Automated Email Reports**: Beautifully designed HTML performance snapshots sent to users, including motivational feedback and growth metrics.
- **Risk Prediction**: Algorithms to identify attendance patterns and predict potential punctuality risks.

### System Management
- **Dockerized Environment**: Fully containerized multi-service architecture (App, Postgres, Redis).
- **Instance Lifecycle Manager**: PID-based locking system to prevent resource conflicts and redundant process spawning.
- **Optimized Builds**: Multi-stage Docker builds with significantly reduced context transfer sizes.

## 3. Technology Stack

### Languages
- **Python 3.10**: Primary backend language for API, business logic, and computer vision integration.
- **JavaScript (ES6+)**: Frontend logic, real-time polling, and interactive chart rendering.
- **Rust**: Used for the high-performance analytics core (`cviaar-analytics`) to ensure fast report generation.
- **HTML5 & CSS3**: Modern responsive layouts using Bootstrap 5 and custom CSS.
- **SQL**: Database querying and management for PostgreSQL.

### Backend Frameworks & Libraries
- **FastAPI**: High-performance web framework for the main application API.
- **SQLAlchemy**: ORM for database management and relationship mapping.
- **OpenCV**: Core library for image processing and facial recognition.
- **MediaPipe**: Used for high-fidelity facial landmark detection.
- **Uvicorn**: ASGI server for running the FastAPI application.
- **Pandas**: Data manipulation and analysis for the Python-side analytics engine.

### Frontend Tools
- **D3.js**: Data-driven document manipulation for the Weekly Attendance chart.
- **Plotly.js**: Interactive charts for attendance velocity and status distribution.
- **Bootstrap 5**: Responsive UI components and layout grid.
- **Jinja2**: Template engine for server-side HTML rendering.

### Infrastructure & DevOps
- **Docker & Docker Compose**: Orchestration of the application, database, and cache services.
- **PostgreSQL 15**: Primary relational database for user data and attendance logs.
- **Redis 7**: In-memory data store for real-time status broadcasting and caching.
- **Uvicorn/Gunicorn**: Production-grade server execution.

## 4. Key Improvements
- **Security**: Robust liveness detection and boundary validation during capture.
- **Performance**: Rust-accelerated analytics and TTLCache implementation for dashboard responsiveness.
- **Scalability**: Multi-container architecture ready for production deployment.
- **User Experience**: Redesigned administrative portal and automated motivational email reporting.
