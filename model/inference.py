import sys
import os
# Add project root to sys.path for absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
import json
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time

# Import enhanced components
from src.feature_extraction.enhanced_extractor import EnhancedFeatureExtractor
from src.models.tcn import TemporalConvNet
from src.models.attention import CrossModalAttention
from src.config_manager import config
from src.logger import get_logger

# Legacy imports for backward compatibility
try:
    from model import AnomalyClassifier
except ImportError:
    AnomalyClassifier = None

logger = get_logger(__name__)

class EnhancedAnomalyDetector(torch.nn.Module):
    """Enhanced anomaly detector with temporal processing and attention."""
    
    def __init__(self, visual_dim: int = 1000, audio_dim: int = 40, 
                 hidden_dim: int = 256, num_classes: int = 2,
                 sequence_length: int = 16, tcn_channels: List[int] = None):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Default TCN channels
        if tcn_channels is None:
            tcn_channels = [64, 128, 256]
        
        # Temporal processing
        self.visual_tcn = TemporalConvNet(visual_dim, tcn_channels)
        self.audio_tcn = TemporalConvNet(audio_dim, tcn_channels)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            tcn_channels[-1], tcn_channels[-1], hidden_dim
        )
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
        # Confidence calibration
        self.temperature = torch.nn.Parameter(torch.ones(1))
        
    def forward(self, visual_seq: torch.Tensor, audio_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Ensure input tensors are on the same device and dtype as model weights
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        visual_seq = visual_seq.to(device=device, dtype=dtype)
        audio_seq = audio_seq.to(device=device, dtype=dtype)

        """Forward pass with temporal sequences.
        Args:
            visual_seq: Visual sequence (batch, seq_len, visual_dim)
            audio_seq: Audio sequence (batch, seq_len, audio_dim)
        Returns:
            Dictionary with logits, probabilities, and attention weights
        """
        batch_size, seq_len, _ = visual_seq.shape

        # Transpose for TCN (batch, features, seq_len)
        visual_tcn_input = visual_seq.transpose(1, 2)
        audio_tcn_input = audio_seq.transpose(1, 2)

        # Temporal processing
        visual_temporal = self.visual_tcn(visual_tcn_input)  # (batch, channels, seq_len)
        audio_temporal = self.audio_tcn(audio_tcn_input)    # (batch, channels, seq_len)

        # Transpose back for attention (batch, seq_len, channels)
        visual_temporal = visual_temporal.transpose(1, 2)
        audio_temporal = audio_temporal.transpose(1, 2)

        # Cross-modal attention
        visual_attended, audio_attended, attention_weights = self.cross_attention(
            visual_temporal, audio_temporal
        )

        # Global pooling
        visual_pooled = torch.mean(visual_attended, dim=1)  # (batch, hidden_dim)
        audio_pooled = torch.mean(audio_attended, dim=1)    # (batch, hidden_dim)

        # Fusion and classification
        # Ensure pooled features are hidden_dim size
        if visual_pooled.shape[1] != self.classifier[0].in_features // 2:
            visual_pooled = torch.nn.functional.pad(visual_pooled, (0, self.classifier[0].in_features // 2 - visual_pooled.shape[1]))
        if audio_pooled.shape[1] != self.classifier[0].in_features // 2:
            audio_pooled = torch.nn.functional.pad(audio_pooled, (0, self.classifier[0].in_features // 2 - audio_pooled.shape[1]))
        fused = torch.cat([visual_pooled, audio_pooled], dim=1)
        logits = self.classifier(fused)

        # Temperature scaling for calibration
        calibrated_logits = logits / self.temperature
        probabilities = torch.softmax(calibrated_logits, dim=1)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'attention_weights': attention_weights,
            'visual_features': visual_pooled,
            'audio_features': audio_pooled
        }

class InferenceEngine:
    """Enhanced inference engine with temporal processing and confidence calibration."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize inference engine.
        
        Args:
            model_path: Path to model weights
            config_path: Path to configuration file
        """
        self.model_path = model_path or 'model_weights.pth'
        self.config_path = config_path
        self.device = self._get_device()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.feature_extractor = EnhancedFeatureExtractor(self.config.get('feature_extraction', {}))
        self.model = None
        self.label_map = self._load_label_map()
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # Confidence thresholds
        self.confidence_thresholds = self.config.get('confidence_thresholds', {})
        self.default_threshold = self.config.get('default_threshold', 0.3)
        
        logger.info(f"Inference engine initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get appropriate device for inference."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        
        # Default configuration
        return {
            'sequence_length': 16,
            'confidence_thresholds': {},
            'default_threshold': 0.7,
            'feature_extraction': {
                'frame_stride': 5,
                'max_frames_per_video': 1000
            }
        }
    
    def _load_label_map(self) -> Dict[str, int]:
        """Load label mapping."""
        label_map_path = 'label_map.json'
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                return json.load(f)
        else:
            # Default binary classification
            return {'normal': 0, 'anomaly': 1}
    
    def load_model(self) -> bool:
        """Load the anomaly detection model."""
        try:
            num_classes = len(self.label_map)
            sequence_length = self.config.get('sequence_length', 16)
            
            # Try to load enhanced model
            self.model = EnhancedAnomalyDetector(
                visual_dim=1000,
                audio_dim=40,
                hidden_dim=256,
                num_classes=num_classes,
                sequence_length=sequence_length
            )
            
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                
                # Handle different state dict formats
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Try to load enhanced model weights
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded enhanced model weights")
                except Exception as e:
                    logger.warning(f"Could not load enhanced model weights: {e}")
                    # Fallback to legacy model if available
                    if AnomalyClassifier is not None:
                        self.model = AnomalyClassifier(
                            visual_dim=1000, audio_dim=40, 
                            hidden_dim=256, num_classes=num_classes
                        )
                        self.model.load_state_dict(state_dict)
                        logger.info("Loaded legacy model weights")
                    else:
                        logger.error("Could not load any model")
                        return False
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            self.model.to(self.device)
            self.model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def calibrate_confidence(self, probabilities: torch.Tensor, 
                           anomaly_type: str) -> torch.Tensor:
        """Apply confidence calibration based on anomaly type.
        
        Args:
            probabilities: Raw probabilities from model
            anomaly_type: Predicted anomaly type
            
        Returns:
            Calibrated probabilities
        """
        # Simple temperature scaling (can be enhanced)
        threshold = self.confidence_thresholds.get(anomaly_type, self.default_threshold)
        
        # Adjust probabilities based on threshold
        calibrated = probabilities.clone()
        max_prob = torch.max(calibrated, dim=1)[0]
        
        # Apply threshold-based adjustment
        adjustment = torch.where(
            max_prob < threshold,
            max_prob * 0.8,  # Reduce confidence for low-confidence predictions
            max_prob
        )
        
        # Normalize
        calibrated = calibrated * (adjustment / max_prob).unsqueeze(1)
        
        return calibrated

    def predict_video(self, video_path: str, extract_temporal: bool = True) -> Dict[str, any]:
        """Predict anomaly for a video file.
        
        Args:
            video_path: Path to video file
            extract_temporal: Whether to use temporal sequence processing
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Initialize feature extractor if needed
            if not self.feature_extractor.is_initialized:
                if not self.feature_extractor.initialize():
                    raise RuntimeError("Failed to initialize feature extractor")

            # Load model if needed
            if self.model is None:
                if not self.load_model():
                    raise RuntimeError("Failed to load model")

            # Always use temporal feature extraction for reliability
            visual_feat, audio_feat = self._extract_temporal_features(video_path)
            
            # Convert to tensors and strictly enforce type/device
            visual_tensor = torch.tensor(visual_feat)
            if visual_tensor.dim() == 2:
                visual_tensor = visual_tensor.unsqueeze(0)
            visual_tensor = visual_tensor.to(device=self.device, dtype=torch.float32)

            audio_tensor = torch.tensor(audio_feat)
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(device=self.device, dtype=torch.float32)
            
            # Inference
            with torch.no_grad():
                if isinstance(self.model, EnhancedAnomalyDetector):
                    results = self.model(visual_tensor, audio_tensor)
                    probabilities = results['probabilities']
                    attention_weights = results['attention_weights']
                else:
                    # Legacy model
                    features = torch.cat([
                        visual_tensor.mean(dim=1),  # Average temporal features
                        audio_tensor.mean(dim=1)
                    ], dim=1)
                    outputs = self.model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    attention_weights = None
                
                # Get prediction
                confidence, predicted = torch.max(probabilities, 1)
                predicted_label = self.inv_label_map[predicted.item()]
                confidence_score = confidence.item()
                
                # Apply confidence calibration
                calibrated_probs = self.calibrate_confidence(probabilities, predicted_label)
                calibrated_confidence = torch.max(calibrated_probs, 1)[0].item()
            
            processing_time = time.time() - start_time
            
            result = {
                'video_path': video_path,
                'predicted_label': predicted_label,
                'confidence': confidence_score,
                'calibrated_confidence': calibrated_confidence,
                'probabilities': probabilities.cpu().numpy().tolist(),
                'processing_time': processing_time,
                'attention_weights': attention_weights,
                'timestamp': time.time()
            }
            
            logger.info(f"Prediction for {video_path}: {predicted_label} "
                       f"(confidence: {confidence_score:.3f}, "
                       f"calibrated: {calibrated_confidence:.3f}, "
                       f"time: {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting video {video_path}: {e}")
            return {
                'video_path': video_path,
                'predicted_label': 'error',
                'confidence': 0.0,
                'calibrated_confidence': 0.0,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _extract_temporal_features(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract temporal sequence features from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (visual_sequence, audio_sequence)
        """
        sequence_length = self.config.get('sequence_length', 16)
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Extract temporal visual features
        visual_sequence = self.feature_extractor.extract_temporal_sequence(
            frames, sequence_length
        )
        
        # Extract audio features (replicated for sequence)
        audio_feat = self.feature_extractor.extract_audio_features(video_path)
        audio_sequence = np.tile(audio_feat, (sequence_length, 1))
        
        return visual_sequence, audio_sequence
    
    def set_confidence_threshold(self, anomaly_type: str, threshold: float) -> None:
        """Set confidence threshold for specific anomaly type.
        
        Args:
            anomaly_type: Type of anomaly
            threshold: Confidence threshold (0.0 to 1.0)
        """
        self.confidence_thresholds[anomaly_type] = threshold
        logger.info(f"Set confidence threshold for {anomaly_type}: {threshold}")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'model_type': type(self.model).__name__,
            'device': str(self.device),
            'num_classes': len(self.label_map),
            'labels': list(self.label_map.keys()),
            'confidence_thresholds': self.confidence_thresholds
        }

# Legacy function for backward compatibility
def predict_video(video_path: str, model_path: Optional[str] = None, 
                 config_path: Optional[str] = None) -> Tuple[str, float]:
    """Legacy function for backward compatibility.
    
    Args:
        video_path: Path to video file
        model_path: Path to model weights
        config_path: Path to configuration file
        
    Returns:
        Tuple of (predicted_label, confidence)
    """
    engine = InferenceEngine(model_path, config_path)
    result = engine.predict_video(video_path)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return 'error', 0.0
    
    print(f"Prediction for {video_path}: {result['predicted_label']} "
          f"(confidence: {result['confidence']:.3f})")
    
    return result['predicted_label'], result['confidence']

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Video Anomaly Detection Inference')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--model_path', help='Path to model weights')
    parser.add_argument('--config_path', help='Path to configuration file')
    parser.add_argument('--temporal', action='store_true', 
                       help='Use temporal sequence processing')
    parser.add_argument('--threshold', type=float, 
                       help='Confidence threshold for anomaly detection')
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = InferenceEngine(args.model_path, args.config_path)
    
    # Set threshold if provided
    if args.threshold is not None:
        engine.default_threshold = args.threshold
    
    # Run inference
    result = engine.predict_video(args.video_path, extract_temporal=args.temporal)
    
    if 'error' not in result:
        print(f"\nResults for {args.video_path}:")
        print(f"  Predicted: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Calibrated Confidence: {result['calibrated_confidence']:.3f}")
        print(f"  Processing Time: {result['processing_time']:.2f}s")
        
        if result['attention_weights'] is not None:
            print("  Attention weights available for visualization")
    else:
        print(f"Error: {result['error']}")
        sys.exit(1)

