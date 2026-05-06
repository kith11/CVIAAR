import webview
import threading
import sys
import os
from waitress import serve
from app import app, scheduler_run

def start_server():
    """Starts the Waitress server in a background thread."""
    print("Starting Waitress Server for Kiosk...")
    serve(app, host="127.0.0.1", port=5000, threads=6)

if __name__ == '__main__':
    # 1. Start the background scheduler (for auto-logout, garbage collection)
    scheduler_thread = threading.Thread(target=scheduler_run, daemon=True)
    scheduler_thread.start()

    # 2. Start the web server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # 3. Create the Kiosk Window
    # fullscreen=True enables Kiosk mode (no title bar, full screen)
    webview.create_window(
        title="Attendance System Kiosk", 
        url="http://127.0.0.1:5000",
        fullscreen=True
    )

    # 4. Start the GUI loop
    print("Launching Kiosk Interface...")
    webview.start()
