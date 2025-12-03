import cv2
import torch
import numpy as np
import os
import json
import time
import threading
import queue
from collections import deque
from typing import Dict, List, Optional, Tuple
import argparse

# Import enhanced components
from src.stream_management.stream_manager import StreamManager
from src.stream_management.video_buffer import VideoBufferPool
from src.processing.pipeline import ProcessingPipeline
from src.alerts.alert_system import AlertSystem
from inference import InferenceEngine
from src.config_manager import config
from src.logger import get_logger

logger = get_logger(__name__)

class RealTimeAnomalyDetector:
    """Enhanced real-time anomaly detection system."""
    
    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None):
        """Initialize real-time detector.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to model weights
        """
        self.config_path = config_path
        self.model_path = model_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.inference_engine = InferenceEngine(model_path, config_path)
        self.stream_manager = None
        self.alert_system = None
        self.processing_pipeline = None
        
        # Real-time processing parameters
        self.sequence_length = self.config.get('sequence_length', 16)
        self.frame_buffer_size = self.config.get('frame_buffer_size', 32)
        self.processing_interval = self.config.get('processing_interval', 1.0)  # seconds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Frame buffers for temporal processing
        self.frame_buffers = {}
        self.last_processing_time = {}
        
        # Threading
        self.processing_thread = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=100)
        
        logger.info("Real-time anomaly detector initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration."""
        if self.config_path and os.path.exists(self.config_path):
            return config.load_config(self.config_path)
        
        return {
            'sequence_length': 16,
            'frame_buffer_size': 32,
            'processing_interval': 1.0,
            'confidence_threshold': 0.7,
            'display_results': True,
            'save_anomalies': True,
            'alert_enabled': True
        }
    
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize inference engine
            if not self.inference_engine.load_model():
                logger.error("Failed to load inference model")
                return False
            
            # Initialize stream manager
            self.stream_manager = StreamManager(self.config.get('stream_config', {}))
            
            # Initialize alert system if enabled
            if self.config.get('alert_enabled', True):
                self.alert_system = AlertSystem(self.config.get('alert_config', {}))
                self.alert_system.initialize()
            
            # Initialize processing pipeline
            self.processing_pipeline = ProcessingPipeline(self.config.get('processing_config', {}))
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def add_stream(self, stream_url: str, stream_id: Optional[str] = None) -> str:
        """Add a video stream for monitoring.
        
        Args:
            stream_url: URL or path to video stream
            stream_id: Optional stream identifier
            
        Returns:
            Stream ID
        """
        if self.stream_manager is None:
            raise RuntimeError("Stream manager not initialized")
        
        from src.stream_management.stream_types import StreamConfig, StreamType
        
        # Determine stream type
        if stream_url.startswith(('rtsp://', 'http://', 'https://')):
            stream_type = StreamType.RTSP if stream_url.startswith('rtsp://') else StreamType.HTTP
        elif stream_url.isdigit():
            stream_type = StreamType.WEBCAM
            stream_url = int(stream_url)
        else:
            stream_type = StreamType.FILE
        
        stream_config = StreamConfig(
            stream_id=stream_id or f"stream_{len(self.frame_buffers)}",
            source_url=stream_url,
            stream_type=stream_type,
            resolution=(640, 480),
            fps=30,
            audio_enabled=False
        )
        
        stream_id = self.stream_manager.add_stream(stream_config)
        self.frame_buffers[stream_id] = deque(maxlen=self.frame_buffer_size)
        self.last_processing_time[stream_id] = 0
        
        logger.info(f"Added stream {stream_id}: {stream_url}")
        return stream_id
    
    def process_frame(self, stream_id: str, frame: np.ndarray, timestamp: float) -> Optional[Dict]:
        """Process a single frame from a stream.
        
        Args:
            stream_id: Stream identifier
            frame: Video frame
            timestamp: Frame timestamp
            
        Returns:
            Detection result if anomaly detected, None otherwise
        """
        # Add frame to buffer
        if stream_id not in self.frame_buffers:
            self.frame_buffers[stream_id] = deque(maxlen=self.frame_buffer_size)
            self.last_processing_time[stream_id] = 0
        
        self.frame_buffers[stream_id].append((frame, timestamp))
        
        # Check if it's time to process
        current_time = time.time()
        if (current_time - self.last_processing_time[stream_id]) < self.processing_interval:
            return None
        
        self.last_processing_time[stream_id] = current_time
        
        # Get frames for temporal sequence
        buffer = self.frame_buffers[stream_id]
        if len(buffer) < self.sequence_length:
            return None
        
        # Sample frames for sequence
        frames = [item[0] for item in list(buffer)[-self.sequence_length:]]
        
        try:
            # Extract temporal features
            visual_sequence = self.inference_engine.feature_extractor.extract_temporal_sequence(
                frames, self.sequence_length
            )
            
            # Create dummy audio sequence (real-time audio processing can be added)
            audio_sequence = np.zeros((self.sequence_length, 40))
            
            # Convert to tensors
            visual_tensor = torch.tensor(visual_sequence, dtype=torch.float32).unsqueeze(0)
            audio_tensor = torch.tensor(audio_sequence, dtype=torch.float32).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.inference_engine.model, 'forward'):
                    results = self.inference_engine.model(visual_tensor, audio_tensor)
                    probabilities = results['probabilities']
                    attention_weights = results.get('attention_weights')
                else:
                    # Legacy model fallback
                    features = torch.cat([
                        visual_tensor.mean(dim=1),
                        audio_tensor.mean(dim=1)
                    ], dim=1)
                    outputs = self.inference_engine.model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    attention_weights = None
                
                confidence, predicted = torch.max(probabilities, 1)
                predicted_label = self.inference_engine.inv_label_map[predicted.item()]
                confidence_score = confidence.item()
            
            # Check if anomaly detected
            if predicted_label != 'normal' and confidence_score >= self.confidence_threshold:
                detection_result = {
                    'stream_id': stream_id,
                    'timestamp': timestamp,
                    'predicted_label': predicted_label,
                    'confidence': confidence_score,
                    'frame': frame.copy(),
                    'attention_weights': attention_weights
                }
                
                # Send alert if enabled
                if self.alert_system:
                    self._send_alert(detection_result)
                
                logger.info(f"Anomaly detected in {stream_id}: {predicted_label} "
                           f"(confidence: {confidence_score:.3f})")
                
                return detection_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing frame from {stream_id}: {e}")
            return None
    
    def _send_alert(self, detection_result: Dict) -> None:
        """Send alert for detected anomaly.
        
        Args:
            detection_result: Detection result dictionary
        """
        try:
            from src.alerts.alert_types import Alert, AlertSeverity
            
            alert = Alert(
                alert_id=f"{detection_result['stream_id']}_{int(detection_result['timestamp'])}",
                camera_id=detection_result['stream_id'],
                anomaly_type=detection_result['predicted_label'],
                confidence=detection_result['confidence'],
                timestamp=detection_result['timestamp'],
                severity=AlertSeverity.HIGH if detection_result['confidence'] > 0.9 else AlertSeverity.MEDIUM,
                metadata={
                    'frame_available': True,
                    'attention_weights_available': detection_result['attention_weights'] is not None
                }
            )
            
            self.alert_system.send_alert(alert, {'frame': detection_result['frame']})
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def start_single_stream(self, stream_url: str, display: bool = True) -> None:
        """Start monitoring a single stream with display.
        
        Args:
            stream_url: URL or path to video stream
            display: Whether to display video with annotations
        """
        if not self.initialize():
            logger.error("Failed to initialize detector")
            return
        
        # Open video capture
        if stream_url.isdigit():
            cap = cv2.VideoCapture(int(stream_url))
        else:
            cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {stream_url}")
            return
        
        stream_id = "main_stream"
        self.frame_buffers[stream_id] = deque(maxlen=self.frame_buffer_size)
        self.last_processing_time[stream_id] = 0
        
        logger.info(f"Starting real-time anomaly detection for: {stream_url}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from video source")
                    break
                
                timestamp = time.time()
                
                # Process frame
                detection_result = self.process_frame(stream_id, frame, timestamp)
                    # Save screenshot if anomaly detected
                    if detection_result is not None:
                        os.makedirs('screenshots', exist_ok=True)
                        label = detection_result['predicted_label']
                        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))
                        filename = f"screenshots/{stream_id}_{label}_{ts_str}.jpg"
                        cv2.imwrite(filename, frame)
                        logger.info(f"Screenshot saved: {filename}")
                
                # Display results
                if display:
                    display_frame = frame.copy()
                    
                    if detection_result:
                        # Anomaly detected
                        label = detection_result['predicted_label']
                        confidence = detection_result['confidence']
                        color = (0, 0, 255)  # Red for anomaly
                        
                        cv2.putText(display_frame, f"ANOMALY: {label}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(display_frame, f"Confidence: {confidence:.3f}", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        # Normal or processing
                        color = (0, 255, 0)  # Green for normal
                        cv2.putText(display_frame, "NORMAL", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Add timestamp
                    cv2.putText(display_frame, f"Time: {time.strftime('%H:%M:%S')}", 
                               (10, display_frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('Real-time Anomaly Detection', display_frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info("Real-time detection stopped")
    
    def start_multi_stream(self, stream_configs: List[Dict]) -> None:
        """Start monitoring multiple streams.
        
        Args:
            stream_configs: List of stream configuration dictionaries
        """
        if not self.initialize():
            logger.error("Failed to initialize detector")
            return
        
        # Add all streams
        for stream_config in stream_configs:
            self.add_stream(stream_config['url'], stream_config.get('id'))
        
        # Start processing
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        logger.info(f"Started monitoring {len(stream_configs)} streams")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def _processing_loop(self) -> None:
        """Main processing loop for multi-stream monitoring."""
        while self.running:
            try:
                # Process frames from all streams
                for stream_id in list(self.frame_buffers.keys()):
                    # Get latest frames from stream manager
                    # This would integrate with the actual stream manager
                    pass
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def stop(self) -> None:
        """Stop the real-time detector."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        if self.stream_manager:
            self.stream_manager.cleanup()
        
        logger.info("Real-time detector stopped")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Enhanced Real-time Video Anomaly Detection')
    parser.add_argument('--stream', default='0', 
                       help='Video stream URL or camera index (default: 0)')
    parser.add_argument('--model_path', help='Path to model weights')
    parser.add_argument('--config_path', help='Path to configuration file')
    parser.add_argument('--no_display', action='store_true', 
                       help='Disable video display')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for anomaly detection')
    
    args = parser.parse_args()
    
    # Create detector
    detector = RealTimeAnomalyDetector(args.config_path, args.model_path)
    detector.confidence_threshold = args.threshold
    
    # Start detection
    detector.start_single_stream(args.stream, display=not args.no_display)

if __name__ == "__main__":
    main()
