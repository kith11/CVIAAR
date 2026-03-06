# CVIAAR System Architecture & Logic Explanation

This document provides a deep dive into the **Hybrid Architecture** of the CVIAAR Attendance System, based on the Horizontal Architecture Flowchart.

## 1. Architectural Overview

The system uses a **multi-layered hybrid approach** to balance real-time responsiveness with permanent data integrity. It is divided into three primary zones:

### **A. Frontend Layer (Client-side)**
*   **Kiosk UI / Web Browser**: The primary interface for staff interaction.
*   **MediaPipe JS**: Performs client-side "Liveness Extraction." It calculates facial landmarks to detect blinks, ensuring a physical person is present before allowing a login. This offloads heavy computer vision processing from the server.

### **B. Backend Layer (FastAPI Server)**
*   **FastAPI Router**: The "Central Brain." It manages all incoming requests and coordinates between different engines.
*   **FaceEngine / OpenCV**: Receives an image buffer from the router and performs the actual biometric identification using the LBPH (Local Binary Patterns Histograms) algorithm.
*   **AnalyticsEngine**: Aggregates raw historical data from the SQL database into structured JSON formats optimized for visualization.

### **C. Hybrid Data Layer (The Core Innovation)**
This layer solves the "latency vs. persistence" problem by splitting data into two paths:

1.  **Live Layer (Upstash Redis)**:
    *   **Logic**: Uses Redis Sets (`SADD` to add, `SCARD` to count).
    *   **Purpose**: Provides **instantaneous** tracking. When a staff member logs in, they are immediately added to a "Present" set in Redis. The dashboard pulls from here for real-time counts without waiting for the slow SQL database.
2.  **Persistence Layer (Supabase SQL)**:
    *   **Logic**: Uses FastAPI `BackgroundTasks`.
    *   **Purpose**: Handles long-term storage. After the user receives a "Success" message (via the Redis write), the system saves the permanent log to Supabase in the background. This ensures the user never waits for a database write to finish.

---

## 2. Step-by-Step Logic Flow

1.  **Identity Capture**: Staff member faces the Kiosk.
2.  **Liveness Check**: MediaPipe JS detects a blink and sends the landmarks + image to the FastAPI Router.
3.  **Recognition**: FastAPI sends the image to the **FaceEngine**. If a match is found, it returns the **Match ID**.
4.  **Instant Recording**:
    *   FastAPI sends an **SADD** command to **Upstash Redis**.
    *   The UI immediately shows a "Success" notification to the user.
5.  **Background Persistence**: FastAPI triggers an **Async BackgroundTask** to write the attendance record to **Supabase SQL**.
6.  **Analytics Loop**: 
    *   The **AnalyticsEngine** periodically pulls historical logs from SQL.
    *   It sends **JSON Trends** to the **D3.js Charts** on the Dashboard for management review.

---

## 3. Why This Architecture?

*   **Zero-Lag User Experience**: By using Redis for the initial "hit," the system feels 10x faster than traditional database-only apps.
*   **Resilience**: If the SQL database is slow or temporarily unreachable, the **Redis Live Layer** still knows who is in the building, preventing system downtime.
*   **Scalability**: Offloading face detection (MediaPipe) to the client and using asynchronous writes (BackgroundTasks) allows the server to handle many kiosks simultaneously without crashing.
