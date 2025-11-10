"""
Complete real-time inference system for action recognition.
Supports webcam, video files, and batch processing.
"""

import torch
import cv2
import numpy as np
import yaml
import json
import argparse
from pathlib import Path
import time
import logging
from collections import deque
import os

# Import our modules
from models.models import create_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Smooth predictions over time using a sliding window."""
    
    def __init__(self, window_size=5, method='weighted'):
        self.window_size = window_size
        self.method = method
        self.predictions_buffer = deque(maxlen=window_size)
        
        if method == 'weighted':
            # More recent predictions get higher weights
            self.weights = np.exp(np.linspace(0, 1, window_size))
            self.weights = self.weights / self.weights.sum()
    
    def update(self, prediction):
        """Update with new prediction and return smoothed result."""
        self.predictions_buffer.append(prediction)
        
        if len(self.predictions_buffer) == 1:
            return prediction
        
        predictions_array = np.array(list(self.predictions_buffer))
        
        if self.method == 'average':
            return np.mean(predictions_array, axis=0)
        elif self.method == 'weighted':
            current_weights = self.weights[-len(self.predictions_buffer):]
            current_weights = current_weights / current_weights.sum()
            return np.average(predictions_array, axis=0, weights=current_weights)
        else:
            return predictions_array[-1]
    
    def reset(self):
        """Reset the prediction buffer."""
        self.predictions_buffer.clear()


class VideoProcessor:
    """Process video frames for action recognition."""
    
    def __init__(self, sequence_length=16, frame_size=(224, 224)):
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Preprocessing transform (ensure float32)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame."""
        # Resize frame
        frame = cv2.resize(frame, self.frame_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and ensure float32
        frame = frame.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        frame = (frame - self.mean) / self.std
        
        # Convert to tensor format (C, H, W) and ensure float32
        frame_tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)
        
        return frame_tensor
    
    def add_frame(self, frame):
        """Add frame to buffer and return video tensor when ready."""
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        
        if len(self.frame_buffer) == self.sequence_length:
            # Stack frames and add batch dimension
            video_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0)
            return video_tensor.float()  # Ensure float32
        
        return None
    
    def reset(self):
        """Reset the frame buffer."""
        self.frame_buffer.clear()


class ActionRecognitionInference:
    """Complete inference system for action recognition."""
    
    def __init__(self, config_path='config/config.yaml', model_path='checkpoints/best_model.pth'):
        """Initialize the inference system."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load class names
        with open('config/class_names.json', 'r') as f:
            class_data = json.load(f)
            self.class_names = class_data['simple_actions']
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize components
        self.video_processor = VideoProcessor(
            sequence_length=self.config['model']['sequence_length'],
            frame_size=tuple(self.config['model']['input_size'])
        )
        
        self.temporal_smoother = TemporalSmoother(window_size=5, method='weighted')
        
        # Performance tracking
        self.inference_times = []
        
        logger.info("✅ Inference system initialized successfully!")
    
    def _load_model(self, model_path):
        """Load the trained model."""
        if not Path(model_path).exists():
            logger.error(f"❌ Model file not found: {model_path}")
            logger.info("💡 Train a model first using: python src/train.py")
            return None
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            model = create_model(
                model_type='cnn_lstm',
                num_classes=len(self.class_names),
                sequence_length=self.config['model']['sequence_length'],
                hidden_dim=self.config['model']['hidden_dim'],
                num_layers=self.config['model']['num_layers'],
                dropout=self.config['model']['dropout'],
                backbone=self.config['model']['backbone']
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(self.device)
            
            # Log model info
            best_acc = checkpoint.get('best_val_acc', 'Unknown')
            logger.info(f"✅ Model loaded from: {model_path}")
            logger.info(f"📊 Best validation accuracy: {best_acc:.4f}" if isinstance(best_acc, float) else f"📊 Accuracy: {best_acc}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None
    
    def predict(self, video_tensor):
        """Predict action from video tensor."""
        if self.model is None:
            return None, 0.0, None
        
        start_time = time.time()
        
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            outputs = self.model(video_tensor)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # Apply temporal smoothing
            smoothed_probs = self.temporal_smoother.update(probabilities)
            
            # Get prediction
            predicted_idx = np.argmax(smoothed_probs)
            confidence = smoothed_probs[predicted_idx]
            predicted_class = self.class_names[predicted_idx]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return predicted_class, confidence, smoothed_probs
    
    def draw_predictions(self, frame, prediction, confidence, all_probs=None):
        """Draw predictions on frame."""
        h, w = frame.shape[:2]
        
        # Determine color based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.4:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Main prediction text
        main_text = f"{prediction}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            main_text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), 
                     (text_width + 20, text_height + 30), 
                     (0, 0, 0), -1)
        
        # Draw main text
        cv2.putText(frame, main_text, (15, text_height + 20), 
                   font, font_scale, color, thickness)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 15
        bar_x = 15
        bar_y = text_height + 40
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + conf_width, bar_y + bar_height), 
                     color, -1)
        
        # Draw top 3 predictions if available
        if all_probs is not None:
            top_indices = np.argsort(all_probs)[-3:][::-1]
            y_offset = text_height + 70
            
            for i, idx in enumerate(top_indices):
                class_name = self.class_names[idx]
                prob = all_probs[idx]
                
                text = f"{class_name}: {prob:.2f}"
                cv2.putText(frame, text, (15, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def process_webcam(self, camera_id=0, display=True, save_video=False, output_path='output_webcam.mp4'):
        """Process webcam stream for real-time inference."""
        logger.info(f"🎥 Starting webcam inference (Camera {camera_id})...")
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video writer setup
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        
        # Reset components
        self.video_processor.reset()
        self.temporal_smoother.reset()
        
        # Performance tracking
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process frame
                video_tensor = self.video_processor.add_frame(frame)
                
                current_prediction = "Collecting frames..."
                confidence = 0.0
                all_probs = None
                
                if video_tensor is not None:
                    predicted_class, conf, probs = self.predict(video_tensor)
                    if predicted_class:
                        current_prediction = predicted_class
                        confidence = conf
                        all_probs = probs
                
                # Draw predictions
                frame = self.draw_predictions(frame, current_prediction, confidence, all_probs)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    
                    cv2.putText(frame, f'FPS: {fps:.1f}', (frame.shape[1] - 100, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Log performance
                    if self.inference_times:
                        avg_inference = np.mean(self.inference_times[-30:])
                        logger.info(f"📊 FPS: {fps:.1f}, Inference: {avg_inference*1000:.1f}ms, Prediction: {current_prediction}")
                
                # Display frame
                if display:
                    cv2.imshow('Action Recognition - Press q to quit', frame)
                
                # Save frame
                if save_video:
                    out.write(frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f'outputs/screenshot_{int(time.time())}.jpg'
                    cv2.imwrite(screenshot_path, frame)
                    logger.info(f"📸 Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            logger.info("\n⚠️ Webcam inference interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if save_video:
                out.release()
                logger.info(f"📹 Video saved: {output_path}")
            if display:
                cv2.destroyAllWindows()
            
            # Print statistics
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                min_time = np.min(self.inference_times)
                
                logger.info("📊 Performance Statistics:")
                logger.info(f"   Average inference: {avg_time*1000:.2f}ms")
                logger.info(f"   Max inference: {max_time*1000:.2f}ms") 
                logger.info(f"   Min inference: {min_time*1000:.2f}ms")
                logger.info(f"   Theoretical max FPS: {1/avg_time:.1f}")
    
    def process_video(self, input_path, output_path=None):
        """Process video file."""
        logger.info(f"🎬 Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"📊 Video info: {width}x{height}, {fps} FPS, {duration:.1f}s")
        
        # Video writer setup
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset components
        self.video_processor.reset()
        self.temporal_smoother.reset()
        
        frame_count = 0
        predictions_history = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Get prediction
                video_tensor = self.video_processor.add_frame(frame)
                
                current_prediction = "Collecting frames..."
                confidence = 0.0
                all_probs = None
                
                if video_tensor is not None:
                    predicted_class, conf, probs = self.predict(video_tensor)
                    if predicted_class:
                        current_prediction = predicted_class
                        confidence = conf
                        all_probs = probs
                        predictions_history.append({
                            'frame': frame_count,
                            'prediction': predicted_class,
                            'confidence': conf,
                            'timestamp': frame_count / fps
                        })
                
                # Draw predictions
                frame = self.draw_predictions(frame, current_prediction, confidence, all_probs)
                
                # Add progress info
                progress = frame_count / total_frames * 100
                cv2.putText(frame, f'Progress: {progress:.1f}%', 
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Save frame
                if output_path:
                    out.write(frame)
                
                # Log progress
                if frame_count % (fps * 5) == 0:  # Every 5 seconds
                    logger.info(f"📈 Progress: {progress:.1f}% - Current: {current_prediction}")
        
        finally:
            cap.release()
            if output_path:
                out.release()
                logger.info(f"✅ Output video saved: {output_path}")
            
            # Save predictions
            if predictions_history:
                pred_file = Path(input_path).stem + '_predictions.json'
                pred_path = f'outputs/{pred_file}'
                
                with open(pred_path, 'w') as f:
                    json.dump({
                        'input_video': input_path,
                        'total_frames': frame_count,
                        'fps': fps,
                        'duration': duration,
                        'predictions': predictions_history,
                        'class_names': self.class_names
                    }, f, indent=2)
                
                logger.info(f"📊 Predictions saved: {pred_path}")
                
                # Summary statistics
                pred_classes = [p['prediction'] for p in predictions_history]
                from collections import Counter
                class_counts = Counter(pred_classes)
                most_common = class_counts.most_common(1)[0]
                
                logger.info(f"📋 Processing Summary:")
                logger.info(f"   Total predictions: {len(predictions_history)}")
                logger.info(f"   Most common action: {most_common[0]} ({most_common[1]} times)")
                logger.info(f"   Average confidence: {np.mean([p['confidence'] for p in predictions_history]):.3f}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Action Recognition Inference')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Input source: webcam, or path to video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam input')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video')
    
    args = parser.parse_args()
    
    print("🎬 Action Recognition Inference")
    print("=" * 35)
    
    # Create inference system
    inference = ActionRecognitionInference()
    
    if inference.model is None:
        print("❌ Cannot start inference without a trained model!")
        print("💡 Train a model first: python src/train.py")
        return
    
    # Process based on source
    if args.source == 'webcam':
        inference.process_webcam(
            camera_id=args.camera,
            display=not args.no_display,
            save_video=args.save_video,
            output_path=args.output or 'outputs/webcam_output.mp4'
        )
    elif Path(args.source).exists():
        inference.process_video(
            input_path=args.source,
            output_path=args.output
        )
    else:
        print(f"❌ Invalid source: {args.source}")
        print("💡 Use 'webcam' or provide path to video file")


if __name__ == "__main__":
    main()