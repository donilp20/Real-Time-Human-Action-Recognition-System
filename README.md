# Real-Time-Human-Action-Recognition-System🎬 Real-Time Human Action Recognition System
A deep learning system for real-time human action recognition using CNN-LSTM architecture. Built with PyTorch, OpenCV, and Streamlit.

Python
PyTorch
License

🌟 Features

🧠 Deep Learning: CNN-LSTM architecture with ResNet backbone

⚡ Real-time Processing: 15-30 FPS inference on webcam streams

🎯 5 Action Classes: Clapping, waving, walking, sitting, jumping

📱 Web Interface: Beautiful Streamlit dashboard with video upload

📊 Rich Visualizations: Confidence charts, heatmaps, and analytics

🚀 Production Ready: Complete training pipeline with checkpointing

📈 Comprehensive Metrics: Training curves, confusion matrices, F1-scores

🚀 Quick Start
1. Setup Environment
bash
git clone <your-repo-url>
cd real_time_action_recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Create Training Data
bash
# Create synthetic training videos (for immediate testing)
python scripts/prepare_dataset.py

# Verify system setup
python test_system.py
3. Train Model
bash
# Train with synthetic data
python src/train.py

# Monitor training (optional)
# tensorboard --logdir logs
4. Run Inference
bash
# Real-time webcam inference
python src/inference.py --source webcam

# Process video file
python src/inference.py --source input_video.mp4 --output result.mp4

# Launch web demo
streamlit run demos/streamlit_demo.py
📁 Project Structure
text
real_time_action_recognition/
├── 📚 README.md                    # This file
├── 📦 requirements.txt             # Python dependencies
├── 🧪 test_system.py              # System testing
├── config/                        # Configuration files
│   ├── config.yaml                # Main configuration
│   └── class_names.json           # Action class definitions
├── src/                           # Core source code
│   ├── models.py                  # CNN-LSTM architecture
│   ├── dataset.py                 # Video dataset handling
│   ├── train.py                   # Training pipeline
│   └── inference.py               # Real-time inference
├── demos/                         # Web applications
│   └── streamlit_demo.py          # Streamlit interface
├── scripts/                       # Utility scripts
│   └── prepare_dataset.py         # Dataset preparation
├── data/                          # Training data
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
└── outputs/                       # Results and videos
🧠 Model Architecture
CNN-LSTM Design
text
Input Video (16 frames × 224×224 RGB)
    ↓
ResNet18 Feature Extractor
    ↓
Spatial Features (16 × 512)
    ↓
2-Layer LSTM (256 hidden units)
    ↓
Classifier (512 → 128 → 5 classes)
    ↓
Action Prediction + Confidence
Supported Actions
👏 Clapping: Hand clapping motions

👋 Waving: Hand waving gestures

🚶 Walking: Person walking

🪑 Sitting: Sitting down action

🦘 Jumping: Vertical jumping motions

🏋️ Training
Configuration
Edit config/config.yaml to customize training:

text
model:
  backbone: 'resnet18'        # CNN backbone
  num_classes: 5              # Action classes
  sequence_length: 16         # Input frames
  hidden_dim: 256             # LSTM hidden size

training:
  batch_size: 8               # Batch size
  learning_rate: 0.001        # Learning rate
  num_epochs: 20              # Max epochs
  patience: 5                 # Early stopping
Training Process
bash
python src/train.py
Training includes:

✅ Automatic train/validation split

✅ Real-time loss and accuracy tracking

✅ Model checkpointing (best + latest)

✅ Early stopping with patience

✅ Learning rate scheduling

✅ Training curve visualization

✅ Confusion matrix generation

Using Real Data
Replace synthetic videos with real action videos:

Record Videos: 2-5 seconds per action

Organize: Place in data/simple_actions/[action_name]/

Format: MP4, AVI, MOV formats supported

Quantity: 20+ videos per class recommended

🎥 Inference
Real-time Webcam
bash
python src/inference.py --source webcam
Features:

Real-time action recognition

Confidence visualization

FPS monitoring

Temporal smoothing

Screenshot capture (press 's')

Video Processing
bash
python src/inference.py --source video.mp4 --output annotated.mp4
Capabilities:

Batch video processing

Progress monitoring

Prediction export (JSON)

Statistics generation

Web Interface
bash
streamlit run demos/streamlit_demo.py
Dashboard Features:

📤 Drag-and-drop video upload

📊 Interactive confidence charts

🎯 Action distribution analysis

🔥 Probability heatmaps

📋 Detailed prediction tables

📸 Sample frame visualization

📊 Performance
Model Performance
Training Accuracy: ~95% (synthetic data)

Validation Accuracy: ~90% (synthetic data)

Real-time FPS: 15-30 (GPU), 5-10 (CPU)

Model Size: ~45 MB

Parameters: ~11M

Hardware Requirements
Minimum:

Python 3.8+

4GB RAM

CPU: Intel i5 or equivalent

Storage: 2GB free space

Recommended:

Python 3.9+

8GB RAM

GPU: GTX 1060 or better

Storage: 10GB free space

🔧 Configuration
Model Settings
text
model:
  backbone: 'resnet18'          # resnet18, resnet50
  num_classes: 5                # Number of actions
  sequence_length: 16           # Frames per video
  hidden_dim: 256               # LSTM hidden size
  num_layers: 2                 # LSTM layers
  dropout: 0.3                  # Dropout rate
Training Settings
text
training:
  batch_size: 8                 # Batch size
  learning_rate: 0.001          # Learning rate
  num_epochs: 20                # Maximum epochs
  patience: 5                   # Early stopping
  device: 'cuda'                # cuda or cpu
Data Settings
text
data:
  dataset_path: './data/simple_actions'
  train_split: 0.7              # 70% training
  val_split: 0.2                # 20% validation
  test_split: 0.1               # 10% testing
🧪 Testing
System Test
bash
python test_system.py
Tests Include:

✅ Dependency verification

✅ Configuration validation

✅ Model creation

✅ Dataset loading

✅ Inference pipeline

✅ Device capabilities

✅ File structure

Component Tests
bash
# Test individual components
python src/models.py          # Model architecture
python src/dataset.py         # Dataset handling
cd scripts && python prepare_dataset.py  # Data preparation
🔧 Troubleshooting
Common Issues
1. CUDA Out of Memory

text
# Reduce batch size in config.yaml
training:
  batch_size: 4  # or smaller
2. No Training Data

bash
# Create synthetic data
python scripts/prepare_dataset.py
3. Low Performance

bash
# Check system capabilities
python test_system.py

# Use CPU if GPU issues
# Edit config.yaml: device: 'cpu'
4. Import Errors

bash
# Ensure you're in project root
ls src/  # Should show models.py, etc.

# Reinstall dependencies
pip install -r requirements.txt
Performance Tips
Training:

Use GPU for 5-10x speedup

Start with synthetic data for testing

Monitor training curves for overfitting

Use early stopping to prevent overtraining

Inference:

Ensure good lighting for webcam

Close other applications for better FPS

Use smaller batch sizes if memory limited

Consider using CPU if GPU unavailable

📈 Advanced Usage
Custom Actions
Collect Data: Record videos for new actions

Update Config: Add class names to config/class_names.json

Retrain: Run python src/train.py

Model Improvements
Try different backbones (ResNet50, EfficientNet)

Experiment with sequence lengths

Add data augmentation

Use transfer learning

Deployment Options
Docker: Containerize the application

API: Create REST API with Flask/FastAPI

Mobile: Convert to ONNX/TensorRT for mobile

Cloud: Deploy on AWS/GCP/Azure

📚 Technical Details
Key Technologies
PyTorch 2.0+: Deep learning framework

OpenCV: Computer vision and video processing

Streamlit: Interactive web applications

ResNet: Pretrained CNN backbone

LSTM: Temporal sequence modeling

Data Pipeline
Video Loading: Multi-format support (MP4, AVI, MOV)

Frame Extraction: Efficient frame sampling

Preprocessing: Normalization and augmentation

Batching: Optimized data loading

Temporal Modeling: Sequence-based learning

Training Pipeline
Data Splitting: Automatic train/val/test splits

Model Creation: CNN-LSTM architecture

Optimization: AdamW with learning rate scheduling

Validation: Per-epoch model evaluation

Checkpointing: Best model saving

Visualization: Training curves and metrics

🤝 Contributing
We welcome contributions! Here's how to help:

Fork the repository

Create a feature branch: git checkout -b feature-name

Commit changes: git commit -am 'Add feature'

Push to branch: git push origin feature-name

Submit a Pull Request

Development Setup
bash
# Clone your fork
git clone https://github.com/your-username/real-time-action-recognition.git

# Install development dependencies
pip install -e .

# Run tests
python test_system.py
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
PyTorch Team: For the excellent deep learning framework

OpenCV Community: For computer vision tools

Streamlit: For the amazing web app framework

Research Community: For CNN and LSTM architectures

📞 Support
Issues: GitHub Issues

Documentation: This README and inline code comments

Examples: Check the demos/ directory

🎯 Future Enhancements
More Actions: Expand to 50+ action classes

Multi-person: Detect actions for multiple people

Mobile App: React Native companion app

Real-time Alerts: Notification system for specific actions

Cloud Deployment: Scalable cloud infrastructure

⭐ Star this repository if you find it helpful!

Built with ❤️ using PyTorch, OpenCV, and Streamlit
