from ultralytics import YOLO
import numpy as np
import torch
from typing import List, Dict, Optional


class YOLOv9Detector:
    """
    Industry-grade YOLOv9 detection wrapper using Ultralytics interface.
    Supports GPU acceleration, mixed precision, and batch-friendly prediction.
    """

    def __init__(self, model_path: str = "yolov9-c-fire.pt", device: Optional[str] = None):
        """
        Initialize the YOLOv9 model and load weights.

        Args:
            model_path (str): Path to the YOLOv9 weights (.pt) file.
            device (str, optional): 'cuda', 'cpu', or None (auto-select).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        self.model.fuse()
        self.model.eval()

        if self.device == "cuda":
            self.model.model.half()  # Use FP16 for better GPU throughput

        # Optional: warm-up
        self._warm_up()
        print(f"[INFO] YOLOv9Detector initialized on {self.device}.")

    def _warm_up(self):
        """Run a dummy inference to reduce first-frame latency (esp. on GPU)."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(dummy, imgsz=640, device=self.device, verbose=False)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Perform object detection on a single image.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            List[Dict]: List of detection results with label, confidence, and bbox.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("Input image must be a 3-channel BGR image as a numpy array.")

        # Run inference
        results = self.model.predict(image, imgsz=640, device=self.device, verbose=False, stream=True)
        output = []

        for result in results:
            for box in result.boxes:
                label_idx = int(box.cls)
                output.append({
                    "label": self.model.names[label_idx],
                    "confidence": float(box.conf),
                    "box": list(map(float, box.xyxy[0]))
                })

        return output
