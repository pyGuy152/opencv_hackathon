#!/bin/sh

# Install missing dependencies
apt-get update && apt-get install -y libgl1 libglib2.0-0

# Run the application
python3 main.py
