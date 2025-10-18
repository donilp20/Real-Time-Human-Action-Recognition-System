"""
Script to organize UCF101 videos into our action classes.
"""

import os
import shutil
from pathlib import Path
import random

def organize_ucf101_data(ucf101_path, output_path, videos_per_class=20):
    """Organize UCF101 videos into our action classes."""
    
    # Mapping UCF101 classes to our classes
    ucf_to_our_classes = {
        # Jumping actions
        'Basketball': 'jumping',
        'BasketballDunk': 'jumping', 
        'HighJump': 'jumping',
        'JumpingJack': 'jumping',
        'TrampolineJumping': 'jumping',
        'JumpRope': 'jumping',
        
        # Walking actions
        'WalkingWithDog': 'walking',
        
        # Sitting actions (people usually sit while doing these)
        'ApplyEyeMakeup': 'sitting',
        'Typing': 'sitting',
        
        # Waving-like actions
        'TennisSwing': 'waving',
        'GolfSwing': 'waving',
        
        # Standing actions
        'Archery': 'standing',
        'StillRings': 'standing',
        'PushUps': 'standing',  # Start position
    }
    
    output_dir = Path(output_path)
    ucf_dir = Path(ucf101_path)
    
    # Create output directories
    our_classes = ['jumping', 'walking', 'sitting', 'waving', 'standing', 'clapping']
    for class_name in our_classes:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    print("🎬 Organizing UCF101 videos...")
    
    for ucf_class, our_class in ucf_to_our_classes.items():
        ucf_class_dir = ucf_dir / ucf_class
        
        if not ucf_class_dir.exists():
            print(f"⚠️ UCF class not found: {ucf_class}")
            continue
            
        # Get all videos from this UCF class
        videos = list(ucf_class_dir.glob('*.avi'))
        
        if len(videos) == 0:
            print(f"⚠️ No videos found in: {ucf_class}")
            continue
        
        # Randomly select videos
        selected_videos = random.sample(videos, min(videos_per_class, len(videos)))
        
        # Copy to our structure
        for i, video_path in enumerate(selected_videos):
            new_name = f"{our_class}_{ucf_class}_{i:03d}.avi"
            destination = output_dir / our_class / new_name
            
            shutil.copy2(video_path, destination)
            print(f"✅ Copied: {video_path.name} → {our_class}/{new_name}")
    
    print(f"\n🎉 Dataset organization complete!")
    print(f"📁 Output directory: {output_path}")
    
    # Print summary
    for class_name in our_classes:
        count = len(list((output_dir / class_name).glob('*.avi')))
        print(f"📊 {class_name}: {count} videos")

if __name__ == "__main__":
    # Update these paths based on your UCF101 location
    UCF101_PATH = "path/to/your/UCF-101"  # Update this
    OUTPUT_PATH = "data/real_actions"
    
    organize_ucf101_data(UCF101_PATH, OUTPUT_PATH, videos_per_class=25)