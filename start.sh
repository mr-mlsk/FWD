#!/bin/bash

# Update package info and install pip
sudo yum update -y
sudo yum install python3-pip -y

# Activate virtualenv or create one
python3 -m venv ~/env
source ~/env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Export Binance API keys as environment variables before running
# export BINANCE_API_KEY="your_key"
# export BINANCE_API_SECRET="your_secret"

# Run the Python strategy script (assumes it's named strategy.py)
python3 strategy.py
