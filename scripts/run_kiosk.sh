#!/bin/bash

# Navigate to script directory and then to project root
cd "$(dirname "$0")/.."

# Activate Virtual Env
source .venv/bin/activate

# Run Kiosk
echo "Starting Kiosk..."
python3 kiosk.py
