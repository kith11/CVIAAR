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
## 🔧 Recent Fixes & Maintenance (March 2026)

The following critical issues were resolved to ensure system stability and performance:

### 1. **Docker Environment & Dependency Fixes**
- **Mediapipe Stability**: Downgraded `mediapipe` to `0.10.9` in [requirements.txt](file:///c%3A/Users/keith/Downloads/projectCVI3/requirements.txt) to maintain compatibility with the legacy `mediapipe.solutions` API used in the codebase.
- **Unused Libraries**: Removed `openai` from dependencies as it was no longer required for the core biometric engine.
- **Docker Build**: Optimized the [Dockerfile](file:///c%3A/Users/keith/Downloads/projectCVI3/Dockerfile) and [docker-compose.yml](file:///c%3A/Users/keith/Downloads/projectCVI3/docker-compose.yml) for consistent environment builds across different host systems.

### 2. **Codebase Integrity & Merge Conflict Resolution**
- **Face Engine Recovery**: Resolved severe merge conflicts in [face_engine.py](file:///c%3A/Users/keith/Downloads/projectCVI3/modules/face_engine.py) that were causing `SyntaxError` and `IndentationError`.
- **Logic Restoration**: Restored the advanced Eye Aspect Ratio (EAR) calculation and state-machine based blink detection for reliable liveness verification.
- **Model Cleanup**: Removed corrupted [lbph_model.yml](file:///c%3A/Users/keith/Downloads/projectCVI3/data/lbph_model.yml) containing Git conflict markers, allowing the system to initialize correctly. (Note: Model retraining is required via the Admin Dashboard).

### 3. **Environment Cleanup**
- Removed accidental files (e.g., `how c2b68d1 --name-only`) that were polluting the project root and potentially causing build issues.

### 4. **Supabase Sync Engine Improvements**
- **Direct Postgres Syncing**: Added a new, more reliable direct PostgreSQL synchronization method using SQLAlchemy in [sync_engine.py](file:///c%3A/Users/keith/Downloads/projectCVI3/modules/sync_engine.py). This acts as a fallback or replacement for the REST API approach.
- **User Record Syncing**: The engine now automatically ensures that user records exist on the remote database before attempting to sync attendance logs, preventing foreign key violations.
- **Enhanced Connectivity Checks**: Updated the internet connectivity check to use `google.com` via HTTPS for better reliability compared to the previous IP-based check.
- **Improved Error Logging**: Added detailed logging and error reporting throughout the synchronization lifecycle to facilitate easier troubleshooting of connection or authentication issues.

---
*Developed for efficient, secure, and insightful institutional attendance management.*
