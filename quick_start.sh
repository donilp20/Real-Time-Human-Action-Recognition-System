#!/bin/bash
# Quick start script for Action Recognition System

echo "🎬 Action Recognition System - Quick Start"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Test system
echo "🧪 Testing system..."
python test_system.py

# Create dataset
echo "🎬 Creating training dataset..."
python scripts/prepare_dataset.py

# Train model
echo "🏋️ Training model..."
python src/train.py

echo "✅ Setup complete! Ready to use:"
echo "1. Webcam inference: python src/inference.py --source webcam"
echo "2. Web demo: streamlit run demos/streamlit_demo.py"