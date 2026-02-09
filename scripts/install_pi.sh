#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")/.."

echo "=== Installing Attendance System Dependencies on Raspberry Pi ==="

# 1. System Dependencies
echo "Step 1: Updating System and Installing Libraries..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libgtk-3-dev libwebkit2gtk-4.0-dev

# 2. Create Virtual Environment
echo "Step 2: Creating Virtual Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
fi

# 3. Install Python Packages
echo "Step 3: Installing Python Requirements..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
# Note: On Pi, we might need to be flexible with versions
pip install flask flask-sqlalchemy pillow imutils python-dotenv openai pandas waitress pywebview

# MediaPipe on ARM can be tricky. Trying standard install first.
# If this fails, we might need 'mediapipe-rpi4' or similar, but Pi 5 often runs standard wheels.
pip install mediapipe==0.10.9 protobuf==3.20.3

# OpenCV (headless is usually better for server, but we need GUI for Kiosk? 
# No, Kiosk is webview. Backend uses cv2. 
# We use opencv-contrib-python for LBPH)
pip install opencv-contrib-python

echo "=== Installation Complete ==="
echo "You can now run ./run_kiosk.sh"
