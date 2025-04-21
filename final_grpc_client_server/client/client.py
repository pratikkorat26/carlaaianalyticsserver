import time

import cv2
import grpc
import os
import sys
import numpy as np
from collections import deque
from random import randint

# Add parent directory to Python path to import detection_pb2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from protos import detection_pb2, detection_pb2_grpc

# Cache random colors per class label
CLASS_COLORS = {}

def get_color(label: str):
    """Assign a consistent random color to each class label."""
    if label not in CLASS_COLORS:
        CLASS_COLORS[label] = (randint(0, 255), randint(0, 255), randint(0, 255))
    return CLASS_COLORS[label]


def encode_frame(frame: np.ndarray) -> bytes:
    """Encodes a frame as JPEG bytes."""
    success, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if success else None


def decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Decodes JPEG bytes into a numpy frame."""
    frame_np = np.frombuffer(jpeg_bytes, np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    for box in detections:
        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        label = f"{box.label} {box.confidence:.2f}"
        color = get_color(box.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def run_grpc_client(source=0, frame_skip=2, max_buffer_size=30):
    """
    Run the YOLOv9 gRPC client with class-colored bounding boxes and streaming optimization.
    :param source: Webcam index or video file path
    :param frame_skip: Number of frames to skip to reduce load
    :param max_buffer_size: Max frame buffer to prevent memory overload
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Cannot open video stream.")
        return

    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = detection_pb2_grpc.DetectionServiceStub(channel)

        frames = deque(maxlen=max_buffer_size)  # buffer for original frames

        def request_generator():
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Optional frame throttling
                if count % frame_skip != 0:
                    count += 1
                    continue

                encoded = encode_frame(frame)
                if encoded:
                    frames.append(frame)  # store original frame

                    yield detection_pb2.FrameRequest(
                        robot_id="robot_1",
                        timestamp=int(time.time() * 1000),
                        frame=encoded
                    )

                count += 1

        # Stream requests and responses
        response_stream = stub.StreamFrames(request_generator())
        for response in response_stream:
            if frames:
                raw_frame = frames.popleft()
                annotated = draw_detections(raw_frame, response.boxes)
                cv2.imshow("YOLOv9 Detection (Streaming)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # source=0 for webcam; use a file path for video
    run_grpc_client(source="fire.mp4", frame_skip=2)
