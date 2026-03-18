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

## 🐳 Docker Environment

This project includes a complete Dockerized development and production environment. The entire stack (Application, PostgreSQL Database, and Redis Cache) can be orchestrated with a single command.

### 🏗️ Getting Started

1.  **Prerequisites**: Ensure you have [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
2.  **Environment Configuration**: Ensure your `.env` file is present in the root directory. If you wish to use the local Dockerized services (Postgres and Redis) instead of remote ones (Supabase/Upstash), you can leave those variables empty or override them in the `docker-compose.yml` file.
3.  **Start the Stack**:
    ```bash
    docker-compose up --build
    ```
    The application will be accessible at [http://localhost:10000](http://localhost:10000).

### 🛠️ Development Features

- **Live Reloading**: The application service is configured with volume mounts (`.:/app`) and `uvicorn --reload`, meaning changes to your local Python files will trigger an automatic restart within the container.
- **Persistent Data**: 
    - PostgreSQL data is persisted in the `pgdata` volume.
    - Redis data is persisted in the `redisdata` volume.
    - Local SQLite database and biometric files are persisted via a mount to the `./data` directory.
- **Health Checks**: Every service includes built-in health checks. The application will only start once the database and cache are verified as healthy.

### 🔍 Debugging & Maintenance

- **View Logs**:
    ```bash
    docker-compose logs -f app
    ```
- **Access Database**:
    ```bash
    docker exec -it cviaar-db psql -U admin -d cviaar
    ```
- **Access Redis**:
    ```bash
    docker exec -it cviaar-redis redis-cli
    ```
- **Stop the Environment**:
    ```bash
    docker-compose down
    ```
    *Note: Use `docker-compose down -v` if you also want to delete the persistent volumes.*

---
## �️ Instance Lifecycle & Reliability

To ensure system stability and prevent resource conflicts (e.g., multiple processes accessing the camera or SQLite database), the application implements strict **Instance Lifecycle Management**.

### 🚦 Singleton Enforcement
- **PID Locking**: The application creates a `data/app.pid` file on startup. If another instance is already running, the new process will log an error and exit immediately.
- **Auto-Cleanup**: On graceful shutdown, the application automatically removes its PID file using Python's `atexit` module.
- **Reloader Awareness**: The logic is designed to work seamlessly with Uvicorn's `--reload` mode, ensuring the reloader parent and worker processes don't conflict with each other.

### 🧹 Systematic Cleanup
If the system enters a state of "operational chaos" with orphaned processes, use the following procedure:
1.  **Stop Docker Stack**: `docker-compose down --remove-orphans`
2.  **Prune Resources**: `docker system prune -f` (reclaims space and removes unused networks/containers).
3.  **Manual PID Reset**: If the app refuses to start due to a stale lock, manually delete `data/app.pid`.

### 🩺 Monitoring
- **Health Checks**: The [Dockerfile](file:///c%3A/Users/keith/Downloads/projectCVI3/Dockerfile) includes a `HEALTHCHECK` that probes the application every 30 seconds.
- **Docker Status**: Use `docker ps` to monitor the `STATUS` column; it should transition from `starting` to `healthy`.

---
## �🔧 Recent Fixes & Maintenance (March 2026)

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
- **Hybrid Sync Strategy**: Implemented a robust synchronization method that first attempts a direct PostgreSQL connection for performance and reliability, with an automatic fallback to the Supabase REST API if authentication or network issues occur.
- **Automated User Sync**: The engine now proactively ensures all local user records exist on the remote database before syncing attendance logs, eliminating foreign key constraint errors during synchronization.
- **Reliable Connectivity Probes**: Switched the internet availability check from a raw IP probe to a standard HTTPS request to `google.com`, ensuring the sync worker only activates when a stable internet connection is actually present.
- **Enhanced Diagnostics**: Integrated detailed logging throughout the synchronization process to provide clear visibility into sync success rates, batch sizes, and specific error messages for faster troubleshooting.

---
*Developed for efficient, secure, and insightful institutional attendance management.*
