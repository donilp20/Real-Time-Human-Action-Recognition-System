"""
Complete Streamlit web application for action recognition.
Upload videos, get predictions, and visualize results.
"""

import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import yaml
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd

# FIXED: Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.models import create_model
    from inference.inference import VideoProcessor, TemporalSmoother
except ImportError as e:
    st.error(f"Could not import required modules: {e}")
    st.info("Make sure you're running from the project root directory and all dependencies are installed")
    st.stop()


class StreamlitActionRecognition:
    """Streamlit application for action recognition."""
    
    def __init__(self):
        self.setup_page()
        self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize session state variables
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = None
        if 'video_processor' not in st.session_state:
            st.session_state.video_processor = None
        if 'temporal_smoother' not in st.session_state:
            st.session_state.temporal_smoother = None
    
    def setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="Action Recognition System",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stProgress .st-bo {
            background-color: #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">🎬 Real-Time Action Recognition</h1>', unsafe_allow_html=True)
        st.markdown("**Upload a video to recognize human actions using deep learning!**")
    
    def load_config(self):
        """Load configuration and class names."""
        try:
            # Load from parent directory
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
            class_names_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'class_names.json')
            
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            with open(class_names_path, 'r') as f:
                class_data = json.load(f)
                self.class_names = class_data['simple_actions']
            
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            st.stop()
    
    def load_model_to_session_state(self):
        """Load the trained model into session state."""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pth')
        
        if not Path(model_path).exists():
            return False, "No trained model found! Train a model first using: `python src/train.py`"
        
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
            
            # Store in session state
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            # Get model info
            best_acc = checkpoint.get('best_val_acc', 'Unknown')
            st.session_state.model_info = {
                'accuracy': best_acc,
                'backbone': self.config['model']['backbone'],
                'sequence_length': self.config['model']['sequence_length'],
                'classes': len(self.class_names),
                'device': str(self.device)
            }
            
            # Setup inference components in session state
            st.session_state.video_processor = VideoProcessor(
                sequence_length=self.config['model']['sequence_length'],
                frame_size=tuple(self.config['model']['input_size'])
            )
            st.session_state.temporal_smoother = TemporalSmoother(window_size=5, method='weighted')
            
            return True, st.session_state.model_info
            
        except Exception as e:
            return False, f"Error loading model: {e}"
    
    def predict_video_tensor(self, video_tensor):
        """Predict action from video tensor using session state model."""
        if not st.session_state.model_loaded or st.session_state.model is None:
            return None, 0.0, None
        
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device).float()  # Ensure float32
            outputs = st.session_state.model(video_tensor)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # Apply temporal smoothing
            smoothed_probs = st.session_state.temporal_smoother.update(probabilities)
            
            # Get prediction
            predicted_idx = np.argmax(smoothed_probs)
            confidence = smoothed_probs[predicted_idx]
            predicted_class = self.class_names[predicted_idx]
        
        return predicted_class, confidence, smoothed_probs
    
    def extract_frames_from_video(self, video_path, max_frames=None):
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Could not open video file"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        frame_count = 0
        
        # Limit frames to avoid memory issues
        if max_frames and total_frames > max_frames:
            step = total_frames // max_frames
        else:
            step = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % step == 0:
                frames.append(frame)
            
            frame_count += 1
            
            if max_frames and len(frames) >= max_frames:
                break
        
        cap.release()
        
        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'extracted_frames': len(frames)
        }
        
        return frames, video_info
    
    def process_video_frames(self, frames):
        """Process video frames and get predictions."""
        st.session_state.video_processor.reset()
        st.session_state.temporal_smoother.reset()
        
        predictions = []
        processed_frames = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, frame in enumerate(frames):
            # Update progress
            progress = (i + 1) / len(frames)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {i+1}/{len(frames)}")
            
            # Process frame
            video_tensor = st.session_state.video_processor.add_frame(frame)
            
            if video_tensor is not None:
                predicted_class, confidence, all_probs = self.predict_video_tensor(video_tensor)
                
                if predicted_class:
                    predictions.append({
                        'frame': i,
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'probabilities': all_probs
                    })
                    
                    # Store processed frame for visualization
                    if len(processed_frames) < 10:  # Limit stored frames
                        processed_frames.append(frame)
        
        progress_bar.empty()
        status_text.empty()
        
        return predictions, processed_frames
    
    def create_confidence_chart(self, predictions):
        """Create confidence chart over time."""
        if not predictions:
            return None
        
        # Prepare data
        frames = [p['frame'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        actions = [p['prediction'] for p in predictions]
        
        # Create line plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=frames,
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='Frame: %{x}<br>Confidence: %{y:.3f}<br>Action: %{text}<extra></extra>',
            text=actions
        ))
        
        fig.update_layout(
            title='Confidence Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Confidence',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_action_distribution_chart(self, predictions):
        """Create action distribution pie chart."""
        if not predictions:
            return None
        
        # Count actions
        action_counts = {}
        for pred in predictions:
            action = pred['prediction']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(action_counts.keys()),
            values=list(action_counts.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title='Action Distribution',
            height=400
        )
        
        return fig
    
    def show_video_analysis_results(self, predictions, video_info, processed_frames):
        """Display comprehensive video analysis results."""
        if not predictions:
            st.warning("No predictions were made. Video might be too short or corrupted.")
            return
        
        # Summary metrics
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(predictions))
        
        with col2:
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            # Most frequent action
            actions = [p['prediction'] for p in predictions]
            most_common = max(set(actions), key=actions.count)
            st.metric("Most Frequent Action", most_common)
        
        with col4:
            st.metric("Video Duration", f"{video_info['duration']:.1f}s")
        
        # Charts
        st.subheader("📈 Detailed Analysis")
        
        tab1, tab2 = st.tabs(["Confidence Timeline", "Action Distribution"])
        
        with tab1:
            confidence_chart = self.create_confidence_chart(predictions)
            if confidence_chart:
                st.plotly_chart(confidence_chart, use_container_width=True)
        
        with tab2:
            distribution_chart = self.create_action_distribution_chart(predictions)
            if distribution_chart:
                st.plotly_chart(distribution_chart, use_container_width=True)
        
        # Sample frames with predictions
        if processed_frames:
            st.subheader("🖼️ Sample Frames with Predictions")
            
            cols = st.columns(min(len(processed_frames), 5))
            for i, frame in enumerate(processed_frames[:5]):
                with cols[i]:
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Find corresponding prediction
                    pred_idx = min(i * len(predictions) // len(processed_frames), len(predictions) - 1)
                    pred = predictions[pred_idx]
                    
                    st.image(frame_rgb, caption=f"{pred['prediction']} ({pred['confidence']:.2f})", use_column_width=True)
        
        # Detailed results table
        with st.expander("📋 Detailed Prediction Results"):
            df = pd.DataFrame(predictions)
            if 'probabilities' in df.columns:
                df = df.drop('probabilities', axis=1)  # Remove complex column for display
            st.dataframe(df, use_container_width=True)
    
    def run(self):
        """Run the Streamlit application."""
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Model Settings")
            
            # Load model button
            if not st.session_state.model_loaded:
                if st.button("🔄 Load Model", type="primary"):
                    with st.spinner("Loading model..."):
                        success, result = self.load_model_to_session_state()
                        
                        if success:
                            st.success("✅ Model loaded successfully!")
                            st.rerun()  # Refresh to show the loaded state
                        else:
                            st.error(f"❌ {result}")
            
            # Show model status and info
            if st.session_state.model_loaded and st.session_state.model_info:
                st.success("✅ Model loaded successfully!")
                
                # Format accuracy properly
                try:
                    if isinstance(st.session_state.model_info['accuracy'], float):
                        accuracy_str = f"{st.session_state.model_info['accuracy']:.4f}"
                    else:
                        accuracy_str = str(st.session_state.model_info['accuracy'])
                    
                    # Display model info safely
                    st.info(f"""
                    **Model Information:**
                    - Accuracy: {accuracy_str}
                    - Backbone: {st.session_state.model_info['backbone']}
                    - Sequence Length: {st.session_state.model_info['sequence_length']} frames
                    - Classes: {st.session_state.model_info['classes']}  
                    - Device: {st.session_state.model_info['device']}
                    """)
                    
                except Exception as display_error:
                    st.warning(f"Model loaded but display error: {display_error}")
                
                st.success("🟢 Model Ready")
                
                # Add reset button
                if st.button("🔄 Reset Model", help="Reset the model to load again"):
                    st.session_state.model_loaded = False
                    st.session_state.model = None
                    st.session_state.model_info = None
                    st.session_state.video_processor = None
                    st.session_state.temporal_smoother = None
                    st.rerun()
            else:
                st.error("🔴 Model Not Loaded")
            
            st.header("📋 Supported Actions")
            for i, action in enumerate(self.class_names):
                st.write(f"{i+1}. {action.title()}")
        
        # Main content
        st.header("📤 Upload Video for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video showing human actions"
        )
        
        if uploaded_file is not None:
            # Display video
            st.video(uploaded_file)
            
            # Analysis settings
            st.subheader("🔧 Analysis Settings")
            col1, col2 = st.columns(2)
            
            with col1:
                max_frames = st.slider("Maximum frames to process", 50, 500, 200, 50,
                                     help="Limit frames to process for faster analysis")
            
            with col2:
                analyze_button = st.button("🚀 Analyze Video", type="primary", 
                                         disabled=(not st.session_state.model_loaded))
            
            if analyze_button:
                if not st.session_state.model_loaded:
                    st.error("Please load the model first!")
                    return
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner("Extracting frames from video..."):
                        frames, video_info = self.extract_frames_from_video(tmp_path, max_frames)
                    
                    if frames is None:
                        st.error(video_info)  # Error message
                        return
                    
                    st.success(f"✅ Extracted {len(frames)} frames from video")
                    
                    # Process frames
                    with st.spinner("Analyzing video... This may take a moment."):
                        predictions, processed_frames = self.process_video_frames(frames)
                    
                    if predictions:
                        st.success(f"✅ Analysis complete! Made {len(predictions)} predictions.")
                        
                        # Show results
                        self.show_video_analysis_results(predictions, video_info, processed_frames)
                    else:
                        st.warning("No predictions could be made. Video might be too short.")
                
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        
        # Instructions
        st.header("💡 How to Use")
        
        with st.expander("📖 Instructions"):
            st.markdown(f"""
            ### Getting Started:
            1. **Load Model**: Click "Load Model" in the sidebar
            2. **Upload Video**: Choose a video file showing human actions
            3. **Adjust Settings**: Set maximum frames to process
            4. **Analyze**: Click "Analyze Video" to get predictions
            
            ### Tips for Best Results:
            - Use clear videos with good lighting
            - Ensure actions are clearly visible
            - Videos should be 2-10 seconds long
            - Supported actions: {', '.join(self.class_names)}
            
            ### Understanding Results:
            - **Confidence Timeline**: Shows prediction confidence over time
            - **Action Distribution**: Shows proportion of different actions detected
            
            ### Technical Details:
            - Model: CNN-LSTM architecture with ResNet18 backbone
            - Input: {self.config['model']['sequence_length']} frame sequences
            - Processing: Real-time inference with temporal smoothing
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("**Built with ❤️ using PyTorch, OpenCV, and Streamlit**")

def main():
    """Main function to run the Streamlit app."""
    try:
        app = StreamlitActionRecognition()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please ensure you're running from the project root directory and all dependencies are installed.")


if __name__ == "__main__":
    main()