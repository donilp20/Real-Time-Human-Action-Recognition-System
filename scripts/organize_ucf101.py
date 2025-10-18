"""
Script to organize UCF101 videos into our 6 action classes.
Run this script to automatically copy and organize UCF101 videos.
"""

import os
import shutil
from pathlib import Path
import random

def organize_ucf101_dataset():
    """
    Organize UCF101 videos into our action recognition classes.
    """
    
    # =============================================
    # CONFIGURATION - UPDATE THESE PATHS
    # =============================================
    
    # Path to your downloaded UCF101 dataset
    UCF101_ROOT = r"C:\Data\Datasets\UCF-101"
    
    # Output path in your project
    OUTPUT_ROOT = r"C:\Data\Real-Time-Human-Action-Recognition-System\data\real_actions"
    
    # Number of videos per class (recommended: 20-30)
    VIDEOS_PER_CLASS = 25
    
    # =============================================
    # UCF101 TO OUR CLASSES MAPPING
    # =============================================
    
    ucf_to_our_mapping = {
        # JUMPING ACTIONS
        'Basketball': 'jumping',           # Basketball shots
        'BasketballDunk': 'jumping',       # Basketball dunks
        'HighJump': 'jumping',             # High jumping
        'JumpingJack': 'jumping',          # Jumping jacks
        'TrampolineJumping': 'jumping',    # Trampoline jumping
        'JumpRope': 'jumping',             # Jump rope
        'PoleVault': 'jumping',            # Pole vaulting
        
        # WALKING ACTIONS
        'WalkingWithDog': 'walking',       # Walking with dog
        
        # SITTING ACTIONS (activities typically done while sitting)
        'ApplyEyeMakeup': 'sitting',       # Applying makeup (usually sitting)
        'Typing': 'sitting',               # Typing (usually sitting)
        'Shaving': 'sitting',              # Shaving (often sitting)
        
        # WAVING-LIKE ACTIONS (arm movements)
        'TennisSwing': 'waving',           # Tennis swings
        'GolfSwing': 'waving',             # Golf swings
        'BaseballPitch': 'waving',         # Baseball pitching
        'VolleyballSpiking': 'waving',     # Volleyball spiking
        
        # STANDING ACTIONS
        'Archery': 'standing',             # Archery (standing position)
        'StillRings': 'standing',          # Gymnastics rings (standing start)
        'ThrowDiscus': 'standing',         # Discus throwing (standing)
        'JavelinThrow': 'standing',        # Javelin throwing (standing)
    }
    
    # =============================================
    # ORGANIZE VIDEOS
    # =============================================
    
    ucf_root = Path(UCF101_ROOT)
    output_root = Path(OUTPUT_ROOT)
    
    # Verify UCF101 path exists
    if not ucf_root.exists():
        print(f"❌ UCF101 path not found: {UCF101_ROOT}")
        print("Please update the UCF101_ROOT path in the script")
        return False
    
    # Create output directories
    our_classes = ['jumping', 'walking', 'sitting', 'waving', 'standing', 'clapping']
    for class_name in our_classes:
        (output_root / class_name).mkdir(parents=True, exist_ok=True)
    
    print("🎬 Starting UCF101 dataset organization...")
    print(f"📂 Source: {UCF101_ROOT}")
    print(f"📂 Destination: {OUTPUT_ROOT}")
    print(f"🎯 Videos per class: {VIDEOS_PER_CLASS}")
    print("-" * 50)
    
    total_copied = 0
    
    # Process each UCF101 class
    for ucf_class, our_class in ucf_to_our_mapping.items():
        ucf_class_dir = ucf_root / ucf_class
        
        if not ucf_class_dir.exists():
            print(f"⚠️  UCF class directory not found: {ucf_class}")
            continue
        
        # Find all video files in this UCF class
        video_files = list(ucf_class_dir.glob('*.avi'))
        
        if len(video_files) == 0:
            print(f"⚠️  No .avi files found in: {ucf_class}")
            continue
        
        # Randomly select videos
        num_to_select = min(VIDEOS_PER_CLASS, len(video_files))
        selected_videos = random.sample(video_files, num_to_select)
        
        print(f"📹 Processing {ucf_class} → {our_class} ({num_to_select} videos)")
        
        # Copy selected videos
        for i, video_path in enumerate(selected_videos):
            # Create new filename: ourclass_ucfclass_number.avi
            new_filename = f"{our_class}_{ucf_class}_{i+1:03d}.avi"
            destination = output_root / our_class / new_filename
            
            try:
                shutil.copy2(video_path, destination)
                print(f"  ✅ {video_path.name} → {new_filename}")
                total_copied += 1
            except Exception as e:
                print(f"  ❌ Error copying {video_path.name}: {e}")
    
    print("-" * 50)
    print(f"🎉 Organization complete! Total videos copied: {total_copied}")
    
    # Print summary statistics
    print("\n📊 Dataset Summary:")
    print("-" * 30)
    for class_name in our_classes:
        class_dir = output_root / class_name
        video_count = len(list(class_dir.glob('*.avi')))
        print(f"{class_name:>10}: {video_count:>3} videos")
    
    # Special note about clapping
    if (output_root / 'clapping').exists():
        clapping_count = len(list((output_root / 'clapping').glob('*.avi')))
        if clapping_count == 0:
            print(f"\n💡 Note: No clapping videos found in UCF101.")
            print(f"   You may need to record a few clapping videos manually.")
    
    return True

def create_clapping_videos_note():
    """Create a note file for clapping videos."""
    clapping_dir = Path("data/real_actions/clapping")
    note_file = clapping_dir / "README.txt"
    
    with open(note_file, 'w') as f:
        f.write("""
CLAPPING VIDEOS NEEDED
=====================

UCF101 doesn't have clapping videos, so you need to add them manually.

Instructions:
1. Record 20-25 short videos (3-5 seconds each) of people clapping
2. Save them as .avi or .mp4 files in this directory
3. Name them: clapping_001.avi, clapping_002.avi, etc.

Tips for recording:
- Good lighting
- Clear hand clapping motion
- Person fills 50-70% of frame
- Different people/angles for variety

Alternatively:
- Search online for "clapping hands video"
- Download royalty-free clapping videos
- Convert to .avi format if needed
        """)
    
    print(f"📝 Created clapping instructions: {note_file}")

if __name__ == "__main__":
    print("🎬 UCF101 Dataset Organization Script")
    print("=" * 40)
    
    # Organize the dataset
    success = organize_ucf101_dataset()
    
    if success:
        # Create note for clapping videos
        create_clapping_videos_note()
        
        print("\n🚀 Next Steps:")
        print("1. Add clapping videos manually (see data/real_actions/clapping/README.txt)")
        print("2. Update config.yaml: dataset_path: './data/real_actions'")
        print("3. Run training: python src/train.py")
    else:
        print("\n❌ Organization failed. Please check the UCF101 path and try again.")