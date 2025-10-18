"""
Optimize UCF101 videos for faster training.
Reduces resolution, duration, and file size.
"""

import os
import cv2
from pathlib import Path
import time
from tqdm import tqdm

def optimize_video(input_path, output_path, target_fps=15, target_duration=3, target_size=(224, 224)):
    """Optimize a single video for training."""
    
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        return False
    
    # Get original properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling
    target_frames = target_fps * target_duration
    frame_step = max(1, total_frames // target_frames)
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, target_size)
    
    frame_count = 0
    saved_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames (skip frames to reduce duration)
        if frame_count % frame_step == 0 and saved_frames < target_frames:
            # Resize frame
            frame_resized = cv2.resize(frame, target_size)
            out.write(frame_resized)
            saved_frames += 1
        
        frame_count += 1
        
        # Stop if we have enough frames
        if saved_frames >= target_frames:
            break
    
    cap.release()
    out.release()
    
    return True

def optimize_dataset():
    """Optimize all videos in real_actions dataset."""
    
    input_dir = Path("data/real_actions")
    output_dir = Path("data/real_actions_optimized")
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output structure
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            (output_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Optimizing videos for faster training...")
    
    total_videos = 0
    optimized_videos = 0
    
    # Process each class
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        video_files = list(class_dir.glob('*.avi')) + list(class_dir.glob('*.mp4'))
        
        print(f"\n📁 Processing {class_name}: {len(video_files)} videos")
        
        for video_file in tqdm(video_files, desc=f"Optimizing {class_name}"):
            output_file = output_dir / class_name / f"{video_file.stem}_opt.mp4"
            
            if output_file.exists():
                continue  # Skip if already optimized
            
            success = optimize_video(
                video_file, 
                output_file,
                target_fps=15,        # Reduced FPS
                target_duration=3,    # 3 second clips
                target_size=(224, 224)  # Standard input size
            )
            
            if success:
                optimized_videos += 1
            
            total_videos += 1
    
    print(f"\n🎉 Optimization complete!")
    print(f"📊 Total videos: {total_videos}")
    print(f"✅ Optimized: {optimized_videos}")
    print(f"📁 Output: {output_dir}")
    
    # Update config to use optimized dataset
    print(f"\n💡 Update config.yaml:")
    print(f"   dataset_path: './data/real_actions_optimized'")

if __name__ == "__main__":
    optimize_dataset()