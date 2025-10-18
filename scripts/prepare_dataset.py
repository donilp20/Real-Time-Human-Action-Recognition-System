"""
Dataset preparation script - creates synthetic videos and downloads real datasets.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import time

def create_synthetic_dataset():
    """Create synthetic training videos for immediate testing."""

    print("🎬 Creating Synthetic Training Dataset")
    print("=" * 40)

    # Ensure data directory exists
    data_dir = Path("data/simple_actions")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Action definitions with motion patterns
    actions = {
        'clapping': {
            'color': (0, 255, 255),  # Yellow
            'description': 'Two circles clapping together'
        },
        'waving': {
            'color': (255, 0, 255),  # Magenta  
            'description': 'Circle waving horizontally'
        },
        'walking': {
            'color': (0, 255, 0),    # Green
            'description': 'Circle walking horizontally'
        },
        'sitting': {
            'color': (255, 0, 0),    # Blue
            'description': 'Circle sitting down'
        },
        'jumping': {
            'color': (0, 0, 255),    # Red
            'description': 'Circle jumping vertically'
        }
    }

    # Video properties
    width, height = 224, 224
    fps = 25
    duration = 3  # seconds
    total_frames = fps * duration
    videos_per_class = 10  # Create 10 videos per class for better training

    total_videos_created = 0

    for action_name, action_info in actions.items():
        action_dir = data_dir / action_name
        action_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📹 Creating {action_name} videos...")

        for video_idx in range(videos_per_class):
            output_path = action_dir / f"synthetic_{action_name}_{video_idx:03d}.mp4"

            # Skip if already exists
            if output_path.exists():
                print(f"  ⏭️  Skipping existing: {output_path.name}")
                continue

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Add variation to make videos different
            variation_speed = np.random.uniform(0.8, 1.2)
            variation_amplitude = np.random.uniform(0.8, 1.2)
            noise_level = np.random.uniform(0, 0.1)

            for frame_idx in range(total_frames):
                # Create frame with dark background
                frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Add some random background noise
                if noise_level > 0:
                    noise = np.random.randint(0, int(255 * noise_level), (height, width, 3), dtype=np.uint8)
                    frame = cv2.add(frame, noise)

                # Time progress
                t = frame_idx / total_frames

                # Generate motion based on action with variations
                if action_name == 'clapping':
                    # Two circles moving together and apart
                    center_y = height // 2 + int(10 * np.sin(t * 3 * np.pi))  # slight vertical movement
                    separation = 30 + 20 * np.sin(t * 10 * np.pi * variation_speed) * variation_amplitude
                    x1 = int(width // 2 - separation)
                    x2 = int(width // 2 + separation)
                    cv2.circle(frame, (x1, center_y), 15, action_info['color'], -1)
                    cv2.circle(frame, (x2, center_y), 15, action_info['color'], -1)

                elif action_name == 'waving':
                    # Circle oscillating horizontally
                    x = int(width // 2 + 40 * np.sin(t * 6 * np.pi * variation_speed) * variation_amplitude)
                    y = height // 2 + int(5 * np.cos(t * 4 * np.pi))  # slight vertical movement
                    cv2.circle(frame, (x, y), 20, action_info['color'], -1)

                elif action_name == 'walking':
                    # Circle moving horizontally with bounce
                    x = int(50 + (width - 100) * t)
                    y = int(height // 2 + 10 * np.sin(t * 8 * np.pi * variation_speed) * variation_amplitude)
                    cv2.circle(frame, (x, y), 18, action_info['color'], -1)

                elif action_name == 'sitting':
                    # Circle moving down and expanding
                    x = width // 2 + int(10 * np.sin(t * 2 * np.pi))  # slight horizontal sway
                    y = int(height // 3 + (height // 3) * t * variation_speed)
                    radius = int(15 + 10 * t * variation_amplitude)
                    cv2.circle(frame, (x, y), radius, action_info['color'], -1)

                elif action_name == 'jumping':
                    # Circle bouncing vertically
                    x = width // 2
                    jump_height = abs(np.sin(t * 4 * np.pi * variation_speed)) * variation_amplitude
                    y = int(height - 50 - jump_height * 100)
                    cv2.circle(frame, (x, y), 20, action_info['color'], -1)

                # Add action label
                cv2.putText(frame, action_name.upper(), (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_info['color'], 2)

                # Add frame number for debugging
                cv2.putText(frame, f"{frame_idx:03d}", (width-50, height-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

                out.write(frame)

            out.release()
            total_videos_created += 1
            print(f"  ✅ Created: {output_path.name}")

    print(f"\n🎉 Dataset creation completed!")
    print(f"📊 Total videos created: {total_videos_created}")
    print(f"📁 Dataset location: {data_dir}")

    # Create dataset summary
    summary = {
        'dataset_type': 'synthetic',
        'total_videos': total_videos_created,
        'videos_per_class': videos_per_class,
        'classes': list(actions.keys()),
        'video_properties': {
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration
        },
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(data_dir / 'dataset_info.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"📋 Dataset info saved: {data_dir / 'dataset_info.json'}")
    return total_videos_created


def create_directory_structure():
    """Create the complete directory structure."""

    directories = [
        'data/simple_actions/clapping',
        'data/simple_actions/waving',
        'data/simple_actions/walking', 
        'data/simple_actions/sitting',
        'data/simple_actions/jumping',
        'checkpoints',
        'logs',
        'outputs',
        'plots'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Directory structure created")


def main():
    """Main function."""
    print("🚀 Dataset Preparation for Action Recognition")
    print("=" * 50)

    # Create directories
    create_directory_structure()

    # Create synthetic dataset
    total_videos = create_synthetic_dataset()

    print(f"\n🎯 Setup Complete!")
    print(f"✅ Created {total_videos} synthetic training videos")
    print(f"📂 Data ready at: data/simple_actions/")

    print("\n🚀 Next Steps:")
    print("1. Run training: python src/train.py")
    print("2. Test inference: python src/inference.py --source webcam")
    print("3. Launch web demo: streamlit run demos/streamlit_demo.py")


if __name__ == "__main__":
    main()