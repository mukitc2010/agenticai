#!/bin/bash

# LLM Fine-tuning Pipeline Script
# This script runs the complete pipeline: setup, data prep, training, testing, merging, and Ollama prep

set -e  # Exit on any error

echo "========================================"
echo "LLM Fine-tuning Pipeline"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Prepare dataset
print_status "Preparing dataset..."
python src/dataset_utils.py

# Train the model
print_status "Starting LoRA fine-tuning..."
python src/train_lora.py

# Test inference
print_status "Testing inference..."
python src/test_inference.py

# Merge LoRA adapters
print_status "Merging LoRA adapters..."
python src/export_model.py

# Prepare for Ollama
print_status "Preparing for Ollama..."
python src/ollama_prep.py

print_status "Pipeline completed successfully!"
print_status "Follow the Ollama commands printed above to run your model."

# Deactivate virtual environment
deactivate