# CVIAAR: Biometric Attendance & Analytical Reasoning System

CVIAAR (Computer Vision Intelligence Attendance & Analytical Reasoning) is a state-of-the-art attendance tracking system that leverages **Edge Computing** for local biometric processing and a **Cloud-Based Dashboard** for remote administration and deep data analysis.

## 🚀 System Architecture: The Edge Computing Advantage

This application is designed with an **Edge-to-Cloud** architecture:
- **Local Kiosk (Edge Device)**: Handles real-time face detection, liveness verification (blink detection), and biometric recognition. Processing happens locally on the device to ensure zero-latency and offline reliability.
- **Online Admin Dashboard**: Hosted at [https://cviaar.onrender.com/](https://cviaar.onrender.com/), providing administrators with the ability to monitor logs, manage staff, and view analytical reports from anywhere in the world.
- **Hybrid Synchronization**: Data captured at the Edge is automatically synchronized with the Cloud database when an internet connection is available, ensuring no data loss.

## ✨ Key Features

### 👤 Advanced Biometric Verification
- **Face Recognition**: Uses high-performance LBPH (Local Binary Patterns Histograms) for staff identification.
- **Liveness Detection**: Integrated blink detection to prevent spoofing using photographs or videos.
- **Staff Portal**: Secure identification via 6-digit Staff ID for personal record access.

### 📊 Predictive Analytics & Reasoning
- **AI Insights**: Integrated AI Chatbot powered by Llama-3.1 for natural language querying of attendance data.
- **Risk Prediction**: Automatically identifies staff at risk of absenteeism or tardiness using historical trends (Late Rate, Absent Rate).
- **Automated Reporting**: Sends beautifully formatted GUI analytical reports to staff's Gmail every 30 days.
- **Early Requests**: Staff can request instant early reports via the Staff Portal.

### 🛡️ Administrative Controls
- **Live Monitor**: Real-time visualization of biometric tracking.
- **Identity Retraining**: Easily reset and recapture face models for staff.
- **Audit Logs**: Track every manual edit or adjustment made to attendance records for transparency.

## 🛠️ Technology Stack

- **Backend**: Python (Flask), SQLAlchemy, SQLite (Edge), PostgreSQL (Cloud/Supabase).
- **Computer Vision**: OpenCV, MediaPipe.
- **Frontend**: Bootstrap 5, D3.js (Data Visualization).
- **AI/ML**: OpenAI SDK (Llama-3.1 integration), LBPH Face Recognizer.
- **Deployment**: Docker, Docker Compose, Gunicorn.

## 📥 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd projectCVI3
   ```

2. **Set up Environment Variables**:
   Create a `.env` file in the root directory and configure the following:
   - `HF_TOKEN`: HuggingFace token for AI Chatbot.
   - `MAIL_USERNAME` & `MAIL_PASSWORD`: Gmail credentials for report sending.
   - `SUPABASE_DB_URL`: Connection string for the Cloud database.

3. **Deploy using Docker**:
   ```bash
   docker-compose up --build -d
   ```

## 🌐 Deployment Links
- **Online Dashboard**: [https://cviaar.onrender.com/](https://cviaar.onrender.com/)

---
*Developed for efficient, secure, and insightful institutional attendance management.*
