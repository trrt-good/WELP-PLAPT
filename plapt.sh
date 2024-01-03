#!/bin/bash

# Assuming this script is in the same directory as plapt_cli.py
# Navigate to the script's directory
cd "$(dirname "$0")"

# Activate your Python environment if necessary
# source /path/to/your/venv/bin/activate

# Run your Python script with arguments
python ./plapt_cli.py "$@"
