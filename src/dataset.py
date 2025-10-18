"""
Fixed dataset handler for action recognition.
Includes all necessary imports and functions.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path
import json
import yaml
from typing import List, Tuple, Optional
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoTransforms:
    """Video-specific data augmentation transforms."""
    
    def __init__(self, training=True, frame_size=(224, 224)):
        self.training = training
        self.frame_size = frame_size
        
        if training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(frame_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(frame_size),
                transforms.CenterCrop(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, frame):
        return self.transform(frame)


class ActionRecognitionDataset(Dataset):
    """Dataset for action recognition from video files."""
    
    def __init__(self, data_path: str, class_names: List[str], sequence_length: int = 16,
                 frame_size: Tuple[int, int] = (224, 224), training: bool = True,
                 max_videos_per_class: Optional[int] = None):
        
        self.data_path = Path(data_path)
        self.class_names = class_names
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.training = training
        self.max_videos_per_class = max_videos_per_class
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Video transformations
        self.transform = VideoTransforms(training=training, frame_size=frame_size)
        
        # Load dataset
        self.video_paths, self.labels = self._load_dataset()
        
        logger.info(f"Dataset loaded: {len(self.video_paths)} videos, {len(self.class_names)} classes")
        self._print_dataset_stats()
    
    def _load_dataset(self) -> Tuple[List[str], List[int]]:
        """Load all video files and their labels."""
        video_paths = []
        labels = []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
        
        for class_name in self.class_names:
            class_dir = self.data_path / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            class_videos = []
            
            # Find all video files
            for ext in video_extensions:
                class_videos.extend(list(class_dir.glob(f'*{ext}')))
            
            # Limit videos per class if specified
            if self.max_videos_per_class and len(class_videos) > self.max_videos_per_class:
                class_videos = random.sample(class_videos, self.max_videos_per_class)
            
            # Add to dataset
            for video_path in class_videos:
                video_paths.append(str(video_path))
                labels.append(class_idx)
        
        return video_paths, labels
    
    def _print_dataset_stats(self):
        """Print dataset statistics."""
        logger.info("Dataset Statistics:")
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            logger.info(f"  {class_name}: {count} videos")
    
    def _extract_frames_cv2(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Sample a fixed number of frames from the video."""
        if len(frames) == 0:
            # Return black frames if no frames available
            black_frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            return [black_frame] * self.sequence_length
        
        if len(frames) <= self.sequence_length:
            # If not enough frames, repeat frames
            sampled_frames = frames.copy()
            while len(sampled_frames) < self.sequence_length:
                sampled_frames.extend(frames)
            return sampled_frames[:self.sequence_length]
        
        # Sample frames uniformly
        if self.training:
            # For training, add randomness
            start_idx = random.randint(0, max(0, len(frames) - self.sequence_length))
            indices = list(range(start_idx, start_idx + self.sequence_length))
        else:
            # For validation/test, sample uniformly
            indices = np.linspace(0, len(frames) - 1, self.sequence_length).astype(int)
        
        return [frames[i] for i in indices]
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames to tensor."""
        processed_frames = []
        
        for frame in frames:
            frame_tensor = self.transform(frame)
            processed_frames.append(frame_tensor)
        
        # Stack frames: (sequence_length, channels, height, width)
        video_tensor = torch.stack(processed_frames)
        return video_tensor
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a video sample and its label."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract frames
            frames = self._extract_frames_cv2(video_path)
            
            if len(frames) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.sequence_length
            
            # Sample frames
            sampled_frames = self._sample_frames(frames)
            
            # Preprocess frames
            video_tensor = self._preprocess_frames(sampled_frames)
            
            return video_tensor, label
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            # Return dummy tensor
            dummy_tensor = torch.zeros(self.sequence_length, 3, self.frame_size[0], self.frame_size[1])
            return dummy_tensor, label


def create_data_loaders(data_path: str, class_names: List[str], 
                       batch_size: int = 8, sequence_length: int = 16,
                       frame_size: Tuple[int, int] = (224, 224),
                       train_split: float = 0.7, val_split: float = 0.2,
                       num_workers: int = 0, pin_memory: bool = True,
                       max_videos_per_class: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create training, validation, and test data loaders."""
    
    # Create full dataset
    full_dataset = ActionRecognitionDataset(
        data_path=data_path,
        class_names=class_names,
        sequence_length=sequence_length,
        frame_size=frame_size,
        training=True,
        max_videos_per_class=max_videos_per_class
    )
    
    if len(full_dataset) == 0:
        logger.error("No videos found in dataset!")
        return None, None, None
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_sample_videos():
    """Create sample synthetic videos."""
    print("🎬 Creating synthetic training videos...")
    
    # Ensure data directory exists
    data_dir = Path("data/real_actions")
    
    actions = {
        'standing': {
            'color': (255, 255, 0),  # Yellow
            'description': 'Stationary upright figure'
        },
        'clapping': {
            'color': (0, 255, 255),  # Cyan
            'description': 'Two hands clapping rhythm'
        },
        'waving': {
            'color': (255, 0, 255),  # Magenta
            'description': 'Hand waving side to side'
        },
        'walking': {
            'color': (0, 255, 0),    # Green
            'description': 'Figure walking with arm swing'
        },
        'sitting': {
            'color': (255, 0, 0),    # Blue
            'description': 'Figure in sitting position'
        },
        'jumping': {
            'color': (0, 0, 255),    # Red
            'description': 'Vertical jumping motion'
        }
    }
    
    width, height = 224, 224
    fps = 25
    duration = 3
    total_frames = fps * duration
    videos_per_class = 8
    
    for action_name, action_info in actions.items():
        action_dir = data_dir / action_name
        action_dir.mkdir(parents=True, exist_ok=True)
        
        for video_idx in range(videos_per_class):
            output_path = action_dir / f"synthetic_{action_name}_{video_idx:03d}.mp4"
            
            if output_path.exists():
                continue
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            variation = random.uniform(0.8, 1.2)
            
            for frame_idx in range(total_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (20, 20, 20)
                
                t = frame_idx / total_frames
                
                if action_name == 'clapping':
                    center_y = height // 2
                    separation = 30 + 20 * np.sin(t * 10 * np.pi * variation)
                    x1 = int(width // 2 - separation)
                    x2 = int(width // 2 + separation)
                    cv2.circle(frame, (x1, center_y), 15, action_info['color'], -1)
                    cv2.circle(frame, (x2, center_y), 15, action_info['color'], -1)
                
                elif action_name == 'waving':
                    x = int(width // 2 + 40 * np.sin(t * 6 * np.pi * variation))
                    y = height // 2
                    cv2.circle(frame, (x, y), 20, action_info['color'], -1)
                
                elif action_name == 'walking':
                    x = int(50 + (width - 100) * t)
                    y = int(height // 2 + 10 * np.sin(t * 8 * np.pi * variation))
                    cv2.circle(frame, (x, y), 18, action_info['color'], -1)
                
                elif action_name == 'sitting':
                    x = width // 2
                    y = int(height // 3 + (height // 3) * t * variation)
                    radius = int(15 + 10 * t)
                    cv2.circle(frame, (x, y), radius, action_info['color'], -1)
                
                elif action_name == 'jumping':
                    x = width // 2
                    jump_height = abs(np.sin(t * 4 * np.pi * variation))
                    y = int(height - 50 - jump_height * 100)
                    cv2.circle(frame, (x, y), 20, action_info['color'], -1)
                
                cv2.putText(frame, action_name.upper(), (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_info['color'], 2)
                
                out.write(frame)
            
            out.release()
            print(f"  ✅ Created: {output_path.name}")
    
    print(f"\n🎉 Created {len(actions) * videos_per_class} synthetic videos!")
    return len(actions) * videos_per_class


if __name__ == "__main__":
    # Test the REAL dataset (not synthetic)
    print("🧪 Testing REAL Dataset")
    print("=" * 25)
    
    # Load config to get class names and data path
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('config/class_names.json', 'r') as f:
        class_data = json.load(f)
        class_names = class_data['simple_actions']
    
    # Test REAL dataset loading
    dataset = ActionRecognitionDataset(
        data_path=config['data']['dataset_path'],  # This should be 'data/real_actions'
        class_names=class_names,
        sequence_length=config['model']['sequence_length'],
        training=True
    )
    
    print(f"\n📊 Dataset: {len(dataset)} videos loaded")
    print(f"📁 Data path: {config['data']['dataset_path']}")
    
    if len(dataset) > 0:
        video, label = dataset[0]
        print(f"📹 Sample video shape: {video.shape}")
        print(f"🏷️  Sample label: {label} ({class_names[label]})")
        
        # Test data loader
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=config['data']['dataset_path'],
            class_names=class_names,
            batch_size=4,
            sequence_length=config['model']['sequence_length']
        )
        
        if train_loader:
            batch_videos, batch_labels = next(iter(train_loader))
            print(f"📦 Batch shape: {batch_videos.shape}")
            print(f"🏷️  Batch labels: {batch_labels}")
    
    print("\n✅ REAL dataset test completed!")