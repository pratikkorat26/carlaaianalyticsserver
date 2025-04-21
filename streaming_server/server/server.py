import os
import logging
import cv2
import grpc
import numpy as np
from concurrent import futures
from typing import Dict, Iterator, List, Tuple, Optional, Any

from detector import YOLOv9Detector
import detection_pb2
import detection_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("detection_server")

# Type definitions
ColorType = Tuple[int, int, int]
DetectionType = Dict[str, Any]
FrameType = np.ndarray

class DetectionService:
    """Handles object detection and tracking operations."""
    
    def __init__(self, model_path: str, tracker_type: str, device: str):
        """
        Initialize the detection service with model and tracking settings.
        
        Args:
            model_path: Path to the YOLO model file
            tracker_type: Type of tracker to use
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.tracker = YOLOv9Detector(
            model_path=model_path, 
            tracker_type=tracker_type, 
            device=device
        )
        self.track_colors: Dict[int, ColorType] = {}
        logger.info(f"Detection service initialized on {device} with {tracker_type}")
    
    def get_color(self, track_id: int) -> ColorType:
        """
        Get a consistent color for a specific track ID.
        
        Args:
            track_id: Unique identifier for the tracked object
            
        Returns:
            RGB color tuple
        """
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        return self.track_colors[track_id]
    
    def track_objects(self, frame: FrameType) -> List[DetectionType]:
        """
        Track objects in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries
        """
        return self.tracker.track(frame)
    
    def draw_annotations(self, frame: FrameType, detections: List[DetectionType]) -> FrameType:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Original image frame
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            track_id = det["track_id"]
            class_name = det["label"]
            conf = det["confidence"]

            color = self.get_color(track_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated


class FrameProcessor:
    """Handles frame encoding/decoding operations."""
    
    @staticmethod
    def decode_frame(frame_bytes: bytes) -> Optional[FrameType]:
        """
        Decode a frame from bytes.
        
        Args:
            frame_bytes: Encoded frame data
            
        Returns:
            Decoded numpy array or None if decoding fails
        """
        try:
            frame_np = np.frombuffer(frame_bytes, np.uint8)
            return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            return None
    
    @staticmethod
    def encode_frame(frame: FrameType) -> Optional[bytes]:
        """
        Encode a frame to JPEG bytes.
        
        Args:
            frame: Image frame as numpy array
            
        Returns:
            Encoded bytes or None if encoding fails
        """
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes() if success else None
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            return None


class DetectionServicer(detection_pb2_grpc.DetectionServiceServicer):
    """gRPC service implementation for object detection."""
    
    def __init__(self, model_path: str = "yolov9-c-fire.pt", 
                 tracker_type: str = "bytetrack.yaml", 
                 use_gpu: bool = True):
        """
        Initialize the gRPC servicer.
        
        Args:
            model_path: Path to the model weights
            tracker_type: Type of tracker to use
            use_gpu: Whether to use GPU acceleration
        """
        device = "cuda" if use_gpu and self._is_cuda_available() else "cpu"
        self.detection_service = DetectionService(
            model_path=model_path,
            tracker_type=tracker_type,
            device=device
        )
        self.frame_processor = FrameProcessor()
        logger.info(f"Detection servicer initialized with model: {model_path}")
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False

    def StreamFrames(self, request_iterator: Iterator[detection_pb2.FrameRequest], 
                     context) -> Iterator[detection_pb2.AnnotatedResponse]:
        """
        Process a stream of frames from clients.
        
        Args:
            request_iterator: Iterator of frame requests
            context: gRPC context
            
        Yields:
            Annotated responses with detection results
        """
        for req in request_iterator:
            try:
                robot_id = req.robot_id
                timestamp = req.timestamp
                logger.info(f"Received frame from Robot ID: {robot_id}")

                # Prepare empty response in case of failure
                empty_response = detection_pb2.AnnotatedResponse(
                    robot_id=robot_id, 
                    timestamp=timestamp
                )
                
                # Decode the frame
                frame = self.frame_processor.decode_frame(req.frame)
                if frame is None:
                    logger.warning(f"Failed to decode frame from Robot ID: {robot_id}")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("Invalid image data received.")
                    yield empty_response
                    continue

                # Track objects in the frame
                try:
                    detections = self.detection_service.track_objects(frame)
                    logger.info(f"Tracked {len(detections)} objects for Robot ID: {robot_id}")
                except Exception as e:
                    logger.error(f"Tracking failed for Robot ID {robot_id}: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"Tracking error: {str(e)}")
                    yield empty_response
                    continue

                # Draw annotations and encode the result
                annotated_frame = self.detection_service.draw_annotations(frame, detections)
                encoded_annotated_frame = self.frame_processor.encode_frame(annotated_frame)
                
                if encoded_annotated_frame is None:
                    logger.warning(f"Failed to encode annotated frame for Robot ID: {robot_id}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details("Failed to encode annotated frame")
                    yield empty_response
                    continue

                # Build and return the response
                response = detection_pb2.AnnotatedResponse(
                    robot_id=robot_id,
                    timestamp=timestamp,
                    annotated_frame=encoded_annotated_frame
                )

                # Add detected objects to the response
                for det in detections:
                    obj = detection_pb2.TrackedObject(
                        label=det["label"],
                        confidence=float(det["confidence"]),  # Ensure it's a Python float
                        x1=float(det["box"][0]),
                        y1=float(det["box"][1]),
                        x2=float(det["box"][2]),
                        y2=float(det["box"][3]),
                        track_id=int(det["track_id"])
                    )
                    response.objects.append(obj)

                yield response

            except Exception as e:
                logger.exception(f"Unhandled exception during processing: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Server error: {str(e)}")
                yield detection_pb2.AnnotatedResponse(
                    robot_id=getattr(req, 'robot_id', 0), 
                    timestamp=getattr(req, 'timestamp', 0)
                )


def get_server_config() -> Dict[str, Any]:
    """
    Get server configuration from environment variables with defaults.
    
    Returns:
        Dictionary of configuration values
    """
    return {
        "host": os.getenv("YOLO_SERVER_HOST", "[::]"),
        "port": int(os.getenv("YOLO_SERVER_PORT", "50051")),
        "max_workers": int(os.getenv("YOLO_MAX_WORKERS", "10")),
        "model_path": os.getenv("YOLO_MODEL_PATH", "yolo11n.pt"),
        "tracker_type": os.getenv("YOLO_TRACKER_TYPE", "bytetrack.yaml"),
        "use_gpu": os.getenv("YOLO_USE_GPU", "1") == "1",
    }


def serve() -> None:
    """Start the gRPC server and wait for termination."""
    config = get_server_config()
    
    logger.info(f"Starting gRPC server on {config['host']}:{config['port']}")
    logger.info(f"Configuration: {config}")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config['max_workers']),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
    )
    
    detection_servicer = DetectionServicer(
        model_path=config['model_path'],
        tracker_type=config['tracker_type'],
        use_gpu=config['use_gpu']
    )

    detection_pb2_grpc.add_DetectionServiceServicer_to_server(detection_servicer, server)
    server.add_insecure_port(f"{config['host']}:{config['port']}")
    
    try:
        server.start()
        logger.info("Server started and listening")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server gracefully")
        server.stop(grace=5)  # 5 seconds grace period
    except Exception as e:
        logger.exception(f"Server error: {e}")
        server.stop(0)


if __name__ == "__main__":
    serve()
