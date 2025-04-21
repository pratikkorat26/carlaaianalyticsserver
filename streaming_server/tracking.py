from ultralytics import YOLO
import numpy as np

class YOLOTracker:
    def __init__(self, model_path="yolov9-c-fire.pt", tracker_type="bytetrack.yaml", device="cuda"):
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        self.device = device

    def track(self, frame):
        results = self.model.track(
            source=frame, 
            device=self.device, 
            tracker=self.tracker_type, 
            persist=True,  # tracks persist across frames
            verbose=False,
            imgsz=640
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else -1

                detections.append({
                    "track_id": track_id,
                    "label": label,
                    "confidence": conf,
                    "box": coords
                })

        return detections
