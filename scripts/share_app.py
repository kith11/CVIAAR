import os
import sys
from pyngrok import ngrok, conf
from dotenv import load_dotenv

import subprocess

# Load .env from project root (one level up)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def start_tunnel():
    # 0. Kill any existing ngrok processes to prevent conflicts
    print("Cleaning up existing ngrok sessions...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ngrok.exe"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        # Give it a moment to release resources
        import time
        time.sleep(1)
    except Exception:
        pass

    # Check for Auth Token
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        print("\n[ERROR] NGROK_AUTH_TOKEN is missing from your .env file!")
        print("1. Go to https://dashboard.ngrok.com/get-started/your-authtoken")
        print("2. Copy your token.")
        print("3. Paste it into your .env file: NGROK_AUTH_TOKEN=your_token_here")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Configure ngrok
    conf.get_default().auth_token = auth_token
    conf.get_default().region = "us"  # Change to 'eu', 'ap', etc. if needed

    try:
        # Open a HTTP tunnel on the default port 5000
        # This points to where your Waitress server is running
        public_url = ngrok.connect(5000).public_url
        print("\n" + "="*60)
        print(f" GLOBAL ACCESS LINK: {public_url}")
        print("="*60)
        print("\nShare this link to access the app from ANYWHERE (Different WiFi/Data).")
        print("Keep this window open to keep the link active.")
        print("Make sure run_prod.bat is running in another window!")
        print("\nPress Ctrl+C to stop sharing.")
        
        # Keep the script running
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
    except KeyboardInterrupt:
        print(" Shutting down tunnel...")
        ngrok.kill()

if __name__ == "__main__":
    start_tunnel()
