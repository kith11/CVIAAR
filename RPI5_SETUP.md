## CVIAAR Raspberry Pi 5 Setup Guide

This guide explains how to run the CVIAAR attendance kiosk on a Raspberry Pi 5 as a local edge device with camera and face recognition.

The goal is:
- Raspberry Pi 5 acts as the kiosk (camera + recognition + local DB)
- Other devices on the same network access the web UI via the Pi’s IP address

---

### 1. Hardware Requirements

- Raspberry Pi 5 (4 GB or 8 GB RAM recommended)
- 32 GB or larger micro‑SD card
- Stable 5V USB‑C power supply
- Camera:
  - USB webcam (recommended for simplicity), or
  - Official Raspberry Pi Camera (CSI) with ribbon cable
- Wired or Wi‑Fi network connection

---

### 2. Base OS Setup

1. Flash **Raspberry Pi OS 64‑bit (Bookworm)** using Raspberry Pi Imager.
2. On first boot:
   - Complete the initial setup wizard
   - Connect to your Wi‑Fi or LAN
   - Enable SSH (optional but recommended)

3. Update the system:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

---

### 3. Enable and Test the Camera

#### 3.1. For USB Webcam

Most USB webcams will work out of the box using the `v4l2` driver.

Check that the camera is detected:

```bash
ls /dev/video*
```

You should see something like `/dev/video0`.

#### 3.2. For CSI Camera (Official Pi Camera)

Run:

```bash
sudo raspi-config
```

Then:
- Interface Options → enable Camera
- Finish and reboot

After reboot, verify:

```bash
libcamera-hello
```

If you see a preview window, the camera is working.

---

### 4. Install System Dependencies

These packages are needed for Python, OpenCV and MediaPipe:

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  libatlas-base-dev \
  libgl1 \
  libglib2.0-0
```

---

### 5. Clone the Project to the Pi

Choose a directory (for example `/home/pi`):

```bash
cd /home/pi
git clone https://github.com/your-account/projectCVI3.git
cd projectCVI3
```

If you copy the folder via USB instead of git, just make sure it ends up in `/home/pi/projectCVI3`.

---

### 6. Create Python Virtual Environment

From the project directory:

```bash
cd /home/pi/projectCVI3
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

---

### 7. Install Python Dependencies

This project has specific version constraints for compatibility on ARM:

- NumPy `< 2`
- MediaPipe `0.10.9`
- Protobuf `3.20.3`

The `requirements.txt` is already pinned appropriately. Install everything:

```bash
pip install --no-cache-dir -r requirements.txt
```

If installation fails due to build time on the Pi, re‑run the command; the most common cause is temporary network or compilation hiccups.

---

### 8. Environment Configuration

The app reads configuration from environment variables when available.

For a simple kiosk deployment, you can export them in the shell or add them to `~/.bashrc`:

```bash
export SECRET_KEY="change_this_secret"
export ADMIN_PASSWORD="admin123"  # or a more secure password
```

To make these persistent, append them to `~/.bashrc`:

```bash
echo 'export SECRET_KEY="change_this_secret"' >> ~/.bashrc
echo 'export ADMIN_PASSWORD="admin123"' >> ~/.bashrc
```

Then either reboot or run:

```bash
source ~/.bashrc
```

---

### 9. Initialize Database and Face Data Folders

The app uses SQLite and will create the DB and folder structure automatically.

From the project directory with the virtual environment activated:

```bash
cd /home/pi/projectCVI3
source .venv/bin/activate
python app.py
```

On first run it will:
- Create `data/attendance.db`
- Ensure `data/faces/` exists

You can stop it with `Ctrl + C` once you confirm it starts without errors.

---

### 10. Running the Kiosk on Raspberry Pi

For development/testing:

```bash
cd /home/pi/projectCVI3
source .venv/bin/activate
gunicorn --bind 0.0.0.0:5000 app:app
```

Then on the Pi itself (kiosk browser):

- Open: `http://localhost:5000/` for the main camera kiosk

On other devices on the same network:

- `http://<pi-ip>:5000/viewer` for the remote viewer (no camera)
- `http://<pi-ip>:5000/admin` for the admin panel

To find the Pi’s IP:

```bash
hostname -I
```

---

### 11. Optional: Systemd Service for Auto‑Start

To have the kiosk start automatically on boot, you can create a `systemd` service.

Create a unit file:

```bash
sudo nano /etc/systemd/system/cviaar.service
```

Example content:

```ini
[Unit]
Description=CVIAAR Attendance Kiosk
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/projectCVI3
Environment="SECRET_KEY=change_this_secret"
Environment="ADMIN_PASSWORD=admin123"
ExecStart=/home/pi/projectCVI3/.venv/bin/gunicorn --bind 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cviaar.service
sudo systemctl start cviaar.service
```

Check status:

```bash
sudo systemctl status cviaar.service
```

---

### 12. Camera and Performance Tips on Pi 5

- Prefer a **lower resolution** (e.g. 640×480 or 800×600) for smoother recognition.
- Ensure adequate cooling; face recognition is CPU‑intensive.
- If performance is poor:
  - Reduce frame processing rate in `index.html` (increase the interval in `setInterval(processFrame, ...)`)
  - Limit the number of enrolled images per user.

---

### 13. Backup and Updates

- Periodically back up:
  - `data/attendance.db` (attendance logs)
  - `data/faces/` (enrolled faces)
  - `data/lbph_model.yml` (trained face model)

To update the app in the future:

```bash
cd /home/pi/projectCVI3
git pull  # if using git
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
sudo systemctl restart cviaar.service
```

