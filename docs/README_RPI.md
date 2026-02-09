# Migrating to Raspberry Pi 5

The **Raspberry Pi 5 (8GB)** is an excellent choice for this project. It is significantly faster than previous models and can easily handle the Face Recognition (OpenCV) and Face Mesh (MediaPipe) workloads.

## Prerequisites

1.  **OS**: Raspberry Pi OS (Bookworm) 64-bit Desktop.
2.  **Hardware**: Pi 5, Camera Module (or USB Webcam), Fast MicroSD Card (A2 class) or NVMe SSD (recommended for fast boot).

## Installation Steps on Raspberry Pi

1.  **Transfer Files**: Copy this entire project folder to the Pi (e.g., to `/home/pi/projectCVI3`).
2.  **Run Installer**:
    Open a terminal on the Pi, go to the folder, and run:
    ```bash
    chmod +x install_pi.sh
    ./install_pi.sh
    ```
    *(This script handles dependencies, virtual environment, and permissions)*.

3.  **Run the App**:
    ```bash
    ./run_kiosk.sh
    ```

## Performance Notes

-   **Boot Time**: The "few minutes" delay you see on Windows is likely due to loading the AI Models (MediaPipe/TensorFlow) and the "Startup" folder delay. On the Pi 5, this should be reasonably fast, especially if you use a high-quality SD card.
-   **Cooling**: Ensure your Pi 5 has an active cooler (fan), as AI tasks can generate heat.

## Auto-Start on Pi

To make it start automatically on boot:

1.  Open the Autostart configuration:
    ```bash
    mkdir -p ~/.config/autostart
    nano ~/.config/autostart/attendance.desktop
    ```
2.  Paste this content:
    ```ini
    [Desktop Entry]
    Type=Application
    Name=Attendance Kiosk
    Exec=/home/pi/projectCVI3/run_kiosk.sh
    Terminal=false
    ```
3.  Save (Ctrl+O) and Exit (Ctrl+X).
4.  Reboot to test.
