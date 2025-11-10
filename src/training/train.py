"""
Fixed training script for action recognition.
Handles path issues and imports correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import json
import os
import time
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Import our modules (fixed paths)
from models.models import create_model
from data.dataset import create_data_loaders, create_sample_videos

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ActionRecognitionTrainer:
    """Fixed trainer for action recognition models."""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the trainer with configuration."""
        
        # Load configuration (fixed path)
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load class names (fixed path)
        class_names_path = 'config/class_names.json'
        if not os.path.exists(class_names_path):
            logger.error(f"Class names file not found: {class_names_path}")
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")
            
        with open(class_names_path, 'r') as f:
            class_data = json.load(f)
            self.class_names = class_data['simple_actions']
        
        # Update config with actual number of classes
        self.config['model']['num_classes'] = len(self.class_names)
        
        # Setup device with better detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            self.device = torch.device('cpu')
            print(f"⚠️  Using CPU (training will be slower)")
            print(f"💡 Consider using Google Colab for GPU training")
        
        # Create directories
        self._setup_directories()
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        logger.info("✅ Trainer initialized successfully!")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"🎯 Classes: {len(self.class_names)}")
        logger.info(f"📊 Model: {self.config['model']['backbone']} + LSTM")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = ['checkpoints', 'logs', 'outputs', 'plots']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def _create_model(self):
        """Create and initialize the model."""
        model = create_model(
            model_type='cnn_lstm',
            num_classes=self.config['model']['num_classes'],
            sequence_length=self.config['model']['sequence_length'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            backbone=self.config['model']['backbone']
        )
        
        model = model.to(self.device)
        
        # Print model info
        model_info = model.get_model_info()
        logger.info(f"📊 Model Information:")
        for key, value in model_info.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.2f}")
            elif isinstance(value, int):
                logger.info(f"   {key}: {value:,}")
            else:
                logger.info(f"   {key}: {value}")
        
        return model
    
    def _create_optimizer(self):
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-4
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler - FIXED VERSION."""
        return ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    
    def _create_data_loaders(self):
        """Create data loaders."""
        # Ensure we have sample data
        data_path = Path(self.config['data']['dataset_path'])
        if not data_path.exists() or len(list(data_path.glob('*/*.mp4'))) == 0:
            logger.info("🎬 No training data found, creating synthetic videos...")
            create_sample_videos()
        
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=self.config['data']['dataset_path'],
            class_names=self.class_names,
            batch_size=self.config['training']['batch_size'],
            sequence_length=self.config['model']['sequence_length'],
            frame_size=tuple(self.config['model']['input_size']),
            train_split=self.config['data']['train_split'],
            val_split=self.config['data']['val_split'],
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        if train_loader is None:
            raise RuntimeError("Failed to create data loaders!")
        
        logger.info(f"📊 Data loaded:")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            # Move data to device
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()
            
            # Update meters
            losses.update(loss.item(), videos.size(0))
            accuracies.update(accuracy.item(), videos.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return losses.avg, accuracies.avg
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        
        losses = AverageMeter()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(self.val_loader, desc='Validation'):
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                losses.update(loss.item(), videos.size(0))
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return losses.avg, accuracy, all_predictions, all_labels
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'class_names': self.class_names,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'checkpoints/latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            logger.info(f"🌟 New best model saved! Validation accuracy: {val_acc:.4f}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"🚀 Starting training for {self.config['training']['num_epochs']} epochs...")
        
        patience_counter = 0
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_predictions, val_labels = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            logger.info(f"📊 Epoch {epoch+1}/{self.config['training']['num_epochs']} ({epoch_time:.2f}s):")
            logger.info(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"   Best Val Acc: {self.best_val_acc:.4f}")
            logger.info(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                logger.info(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}")


def main():
    """Main training function."""
    print("🎬 Fixed Action Recognition Training")
    print("=" * 40)
    
    try:
        # Create trainer
        trainer = ActionRecognitionTrainer()
        
        # Start training
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        logger.info("🏁 Training session ended")


if __name__ == "__main__":
    main()