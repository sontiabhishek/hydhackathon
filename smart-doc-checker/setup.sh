#!/bin/bash
# This script automates the setup of the Python environment for the summarizer app.
# It creates a virtual environment and installs all necessary dependencies.

set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Setting up Python virtual environment ---"

# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "❌ Error: python3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo "1. Creating virtual environment in '.venv'..."
python3 -m venv .venv
echo "✅ Virtual environment created successfully."

# Install dependencies using the pip from the new virtual environment
echo "2. Installing dependencies from requirements.txt..."
.venv/bin/pip install -r requirements.txt
echo "✅ Dependencies installed successfully."

echo ""
echo "--- Setup Complete ---"
echo "To activate the virtual environment for your current session, run:"
echo "source .venv/bin/activate"
echo ""
echo "Then, to run the application, use the launcher script:"
echo "python launch.py"
echo ""

exit 0