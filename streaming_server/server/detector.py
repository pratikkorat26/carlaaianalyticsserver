# streaming_server/server/detector.py
"""
High-performance object detection and tracking module using the Ultralytics YOLO framework.
Optimized for real-time processing in production environments.
"""

import os
import logging
from pathlib import Path
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple, Any

import numpy as np
import torch
from ultralytics import YOLO

# Configure logging
logger = logging.getLogger("yolo-detector")


class TrackerType(str, Enum):
    """Available object trackers supported by Ultralytics."""
    BYTETRACK = "bytetrack.yaml"
    BOTSORT = "botsort.yaml"


class ModelConfig:
    """Configuration for YOLO model parameters."""

    def __init__(
            self,
            model_path: str = "yolov9-c-fire.pt",
            device: Optional[str] = None,
            imgsz: Union[int, Tuple[int, int]] = 640,
            half_precision: bool = True,
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
    ):
        """
        Initialize model configuration.

        Args:
            model_path: Path to the YOLOv9 model weights
            device: Device to run inference on ("cuda", "cpu", or None for auto)
            imgsz: Input image size (single value or width,height tuple)
            half_precision: Whether to use FP16 precision on GPU
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = self._validate_model_path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.half_precision = half_precision and self.device == "cuda"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @staticmethod
    def _validate_model_path(model_path: str) -> str:
        """
        Validate that the model path exists or points to a valid model ID.

        Args:
            model_path: Path to the model or model identifier

        Returns:
            Validated model path

        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        # Skip validation for built-in models or if it's a URL
        if model_path.startswith(("yolov", "http://", "https://")):
            return model_path

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return str(path)


class YOLOv9Detector:
    """
    Industrial-grade YOLOv9 detector with integrated tracking capabilities.

    Features:
    - Fast initialization with optional warm-up
    - GPU acceleration with mixed precision support
    - Memory-efficient streaming mode
    - Configurable detection parameters
    - Built-in object tracking
    """

    def __init__(
            self,
            model_path: str = "yolov9-c-fire.pt",
            tracker_type: str = "bytetrack",
            device: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the YOLOv9 detector.

        Args:
            model_path: Path to the YOLO model
            tracker_type: Type of tracker to use
            device: Device to run inference on ("cuda", "cpu", or None for auto)
            **kwargs: Additional configuration parameters
        """
        # Create model config
        self.config = ModelConfig(
            model_path=model_path,
            device=device,
            **{k: v for k, v in kwargs.items() if hasattr(ModelConfig.__init__, k)}
        )

        # Validate tracker type
        if isinstance(tracker_type, str):
            try:
                # Handle .yaml extension in tracker name
                clean_tracker = tracker_type.lower().replace('.yaml', '')
                self.tracker_type = TrackerType(clean_tracker)
            except ValueError:
                logger.warning(f"Invalid tracker type: {tracker_type}. Using {TrackerType.BYTETRACK}")
                self.tracker_type = TrackerType.BYTETRACK
        else:
            self.tracker_type = tracker_type

        # Load model
        try:
            logger.info(f"Loading YOLO model from {self.config.model_path} on {self.config.device}")
            self.model = YOLO(self.config.model_path)
            self.model.to(self.config.device)

            # Optimize model
            self.model.fuse()

            # Use half precision if configured
            if self.config.half_precision:
                self.model.model.half()

            # Warm up the model
            self._warm_up()

            logger.info(f"YOLOv9 detector initialized successfully on {self.config.device}")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise

    def _warm_up(self) -> None:
        """
        Run a dummy inference pass to initialize and warm up the model.
        This reduces latency on the first real detection request.
        """
        try:
            logger.debug("Warming up YOLO model...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy,
                imgsz=self.config.imgsz,
                device=self.config.device,
                verbose=False
            )
            logger.debug("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform object detection on a single image.

        Args:
            image: Input image as a numpy array (BGR format, HWC order)

        Returns:
            List of detection results, each as a dictionary with keys:
            - label: Class name
            - class_id: Class index
            - confidence: Detection confidence [0-1]
            - box: Bounding box coordinates [x1, y1, x2, y2]

        Raises:
            ValueError: If the input image format is invalid
            RuntimeError: If detection fails
        """
        # Validate input
        self._validate_image(image)

        try:
            # Run inference with streaming results
            results = self.model.predict(
                image,
                imgsz=self.config.imgsz,
                device=self.config.device,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
                stream=True
            )

            # Process results
            detections = []
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue

                for box in result.boxes:
                    try:
                        # Extract box data
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        xyxy = box.xyxy.cpu().numpy()[0]  # Convert to numpy array

                        # Get class name
                        label = result.names[cls_id]

                        # Create detection dict
                        detection = {
                            "label": label,
                            "class_id": cls_id,
                            "confidence": conf,
                            "box": xyxy.tolist()  # [x1, y1, x2, y2]
                        }
                        detections.append(detection)
                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise RuntimeError(f"Detection failed: {e}")

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform object detection and tracking on a single frame.

        Args:
            frame: Input image as a numpy array (BGR format, HWC order)

        Returns:
            List of tracked objects, each as a dictionary with keys:
            - track_id: Unique tracking ID
            - label: Class name
            - confidence: Detection confidence [0-1]
            - box: Bounding box coordinates [x1, y1, x2, y2]

        Raises:
            ValueError: If the input frame format is invalid
            RuntimeError: If tracking fails
        """
        # Validate input
        self._validate_image(frame)

        try:
            # Run tracking
            results = self.model.track(
                source=frame,
                device=self.config.device,
                tracker=self.tracker_type.value,
                persist=True,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
                imgsz=self.config.imgsz
            )

            # Extract tracking results
            if not results or len(results) == 0:
                return []

            result = results[0]  # Get first result (single image mode)

            # Process tracked objects
            tracked_objects = []
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    try:
                        # Skip boxes without tracking IDs
                        if not hasattr(box, 'id') or box.id is None:
                            continue

                        # Extract box data
                        track_id = int(box.id.item())
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]

                        # Get class name
                        label = result.names[cls_id]

                        # Create tracking dict
                        tracked_obj = {
                            "track_id": track_id,
                            "label": label,
                            "class_id": cls_id,
                            "confidence": conf,
                            "box": xyxy.tolist()  # [x1, y1, x2, y2]
                        }
                        tracked_objects.append(tracked_obj)
                    except Exception as e:
                        logger.warning(f"Error processing tracked object: {e}")

            return tracked_objects

        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            raise RuntimeError(f"Tracking failed: {e}")

    def _validate_image(self, image: np.ndarray) -> None:
        """
        Validate that the input is a proper image.

        Args:
            image: Input image array

        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if image.ndim != 3:
            raise ValueError("Input must be a 3-channel image (HWC format)")

        if image.shape[2] != 3:
            raise ValueError("Input must have 3 channels (BGR)")

        if image.dtype != np.uint8:
            raise ValueError("Input must be an 8-bit unsigned integer array")