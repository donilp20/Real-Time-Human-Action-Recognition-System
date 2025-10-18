"""
Fixed CNN-LSTM model architecture for action recognition.
Addresses the BatchNorm issue with single samples.
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class CNNLSTMActionRecognizer(nn.Module):
    """Fixed CNN-LSTM model for action recognition."""
    
    def __init__(self, num_classes=5, sequence_length=16, hidden_dim=256, 
                 num_layers=2, dropout=0.3, backbone='resnet18'):
        super(CNNLSTMActionRecognizer, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.backbone_name = backbone
        
        # CNN Backbone for spatial feature extraction
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')  # Fixed deprecated warning
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer from ResNet
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # FIXED classifier - removed BatchNorm to avoid single sample issue
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"✅ Model initialized:")
        print(f"   - Backbone: {backbone}")
        print(f"   - Feature dim: {self.feature_dim}")
        print(f"   - LSTM hidden: {hidden_dim}")
        print(f"   - LSTM layers: {num_layers}")
        print(f"   - Classes: {num_classes}")
        print(f"   - Sequence length: {sequence_length}")
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        """Extract CNN features from video frames."""
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape to process all frames together
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features using CNN backbone
        features = self.backbone(x)  # (batch_size * seq_len, feature_dim, 1, 1)
        
        # Flatten spatial dimensions
        features = features.view(features.size(0), -1)  # (batch_size * seq_len, feature_dim)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        return features
    
    def forward(self, x):
        """Forward pass through the model."""
        # Extract spatial features
        spatial_features = self.extract_features(x)  # (batch_size, seq_len, feature_dim)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(spatial_features)
        
        # Use last LSTM output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Classify
        logits = self.classifier(last_output)  # (batch_size, num_classes)
        
        return logits
    
    def get_model_info(self):
        """Get detailed model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': f'CNN-LSTM ({self.backbone_name})',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'feature_extractor': self.backbone_name,
            'temporal_model': 'LSTM',
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim
        }


def create_model(model_type='cnn_lstm', num_classes=5, sequence_length=16, 
                hidden_dim=256, num_layers=2, dropout=0.3, backbone='resnet18'):
    """Factory function to create model."""
    
    if model_type == 'cnn_lstm':
        model = CNNLSTMActionRecognizer(
            num_classes=num_classes,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            backbone=backbone
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing Fixed CNN-LSTM Model")
    print("=" * 35)
    
    # Create model
    model = create_model(
        model_type='cnn_lstm',
        num_classes=5,
        sequence_length=16,
        hidden_dim=256,
        backbone='resnet18'
    )
    
    # Test with single batch (this was causing the error)
    dummy_input = torch.randn(1, 16, 3, 224, 224)  # Single batch
    print(f"📥 Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"📤 Output shape: {output.shape}")
    print(f"📊 Output sample: {output[0].detach().numpy()}")
    
    # Model information
    info = model.get_model_info()
    print(f"\n📋 Model Information:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        elif isinstance(value, int):
            print(f"   {key}: {value:,}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Model test completed successfully!")