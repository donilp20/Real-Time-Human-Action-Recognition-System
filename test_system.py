"""
Comprehensive system test script.
Tests all components and dependencies.
"""

import sys
import os
import torch
import importlib.util
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("📦 Testing Dependencies...")

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM',
        'PIL': 'Pillow',
        'streamlit': 'Streamlit'
    }

    missing = []
    installed = []

    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            elif package == 'sklearn':
                import sklearn
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)

            installed.append(name)
            print(f"  ✅ {name}")

        except ImportError:
            missing.append(name)
            print(f"  ❌ {name} - MISSING")

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All {len(installed)} dependencies installed!")
        return True


def test_configuration():
    """Test configuration files."""
    print("\n⚙️ Testing Configuration...")

    config_files = {
        'config/config.yaml': 'Main configuration',
        'config/class_names.json': 'Class names'
    }

    for file_path, description in config_files.items():
        if Path(file_path).exists():
            try:
                if file_path.endswith('.yaml'):
                    import yaml
                    with open(file_path, 'r') as f:
                        yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    import json
                    with open(file_path, 'r') as f:
                        json.load(f)

                print(f"  ✅ {description}: {file_path}")
            except Exception as e:
                print(f"  ❌ {description}: Invalid format - {e}")
                return False
        else:
            print(f"  ❌ {description}: File not found - {file_path}")
            return False

    print("✅ Configuration files validated!")
    return True


def test_model_creation():
    """Test model creation."""
    print("\n🧠 Testing Model Creation...")

    try:
        # Add src to path
        sys.path.append('src')
        from models.models import create_model

        # Test model creation
        model = create_model(
            model_type='cnn_lstm',
            num_classes=5,
            sequence_length=16,
            hidden_dim=256
        )

        # Test forward pass
        dummy_input = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"  ✅ Model created successfully")
        print(f"  ✅ Forward pass: {dummy_input.shape} -> {output.shape}")

        # Test model info
        info = model.get_model_info()
        print(f"  ✅ Model info: {info['total_parameters']:,} parameters")

        return True

    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False


def test_dataset():
    """Test dataset functionality."""
    print("\n📊 Testing Dataset...")

    try:
        sys.path.append('src')
        from data.dataset import ActionRecognitionDataset

        # Test with dummy directory structure
        class_names = ['clapping', 'waving', 'walking', 'sitting', 'jumping']

        # Create dataset (will work even without videos)
        dataset = ActionRecognitionDataset(
            data_path='data/simple_actions',
            class_names=class_names,
            sequence_length=16,
            training=True
        )

        print(f"  ✅ Dataset created: {len(dataset)} videos found")
        print(f"  ✅ Classes: {len(class_names)}")

        return True

    except Exception as e:
        print(f"  ❌ Dataset test failed: {e}")
        return False


def test_inference_components():
    """Test inference components."""
    print("\n🎥 Testing Inference Components...")

    try:
        sys.path.append('src')
        from inference.inference import VideoProcessor, TemporalSmoother

        # Test video processor
        processor = VideoProcessor(sequence_length=16)

        # Test with dummy frame
        dummy_frame = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8).numpy()
        result = processor.add_frame(dummy_frame)

        print(f"  ✅ VideoProcessor created")
        print(f"  ✅ Frame processing: {dummy_frame.shape} -> buffer size {len(processor.frame_buffer)}")

        # Test temporal smoother
        smoother = TemporalSmoother(window_size=5)
        dummy_prediction = torch.rand(5).numpy()
        smoothed = smoother.update(dummy_prediction)

        print(f"  ✅ TemporalSmoother created")
        print(f"  ✅ Prediction smoothing: {dummy_prediction.shape} -> {smoothed.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Inference components test failed: {e}")
        return False


def test_device_capabilities():
    """Test device capabilities."""
    print("\n🔧 Testing Device Capabilities...")

    # Test CUDA
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  ✅ CUDA version: {torch.version.cuda}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✅ GPU memory: {memory_gb:.1f} GB")
    else:
        print(f"  ⚠️  CUDA not available - will use CPU")

    # Test OpenCV camera access
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"  ✅ Camera access available")
            cap.release()
        else:
            print(f"  ⚠️  Camera not accessible")
    except Exception as e:
        print(f"  ⚠️  Camera test failed: {e}")

    return True


def test_file_structure():
    """Test project file structure."""
    print("\n📁 Testing File Structure...")

    required_files = [
        'src/models.py',
        'src/dataset.py', 
        'src/train.py',
        'src/inference.py',
        'demos/streamlit_demo.py',
        'config/config.yaml',
        'config/class_names.json',
        'requirements.txt'
    ]

    missing_files = []

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files present!")
        return True


def main():
    """Run all tests."""
    print("🧪 Comprehensive System Test")
    print("=" * 40)

    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration), 
        ("Model Creation", test_model_creation),
        ("Dataset", test_dataset),
        ("Inference Components", test_inference_components),
        ("Device Capabilities", test_device_capabilities)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:.<30} {status}")
        if result:
            passed += 1

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! System is ready to use.")
        print(f"\n🚀 Next Steps:")
        print(f"1. Create training data: python scripts/prepare_dataset.py")
        print(f"2. Train model: python src/train.py")
        print(f"3. Test inference: python src/inference.py --source webcam")
        print(f"4. Web demo: streamlit run demos/streamlit_demo.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        print(f"\n🔧 Common fixes:")
        print(f"- Install missing dependencies: pip install -r requirements.txt")
        print(f"- Ensure you're in the project root directory")
        print(f"- Check that all files were created properly")


if __name__ == "__main__":
    main()
