import os
import sys
import cv2
import grpc
import numpy as np
from concurrent import futures
from typing import Iterator

# Add parent directory for detection_pb2 imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detector import YOLOv9Detector
from protos import detection_pb2, detection_pb2_grpc


class DetectionServicer(detection_pb2_grpc.DetectionServiceServicer):
    """
    gRPC Servicer for YOLOv9-based real-time object detection.
    Processes incoming video frames and returns detection results.
    """
    def __init__(self, model_path: str = "yolov9-c-fire.pt", use_gpu: bool = True):
        self.detector = YOLOv9Detector(model_path=model_path)
        print("[INFO] DetectionServicer initialized.")

    def StreamFrames(self, request_iterator: Iterator[detection_pb2.FrameRequest], context) -> Iterator[
        detection_pb2.DetectionResponse]:
        for req in request_iterator:
            try:
                # Log receipt of request
                print(f"[INFO] Received frame from Robot ID: {req.robot_id}, Timestamp: {req.timestamp}")

                # Decode the image from bytes
                frame_np = np.frombuffer(req.frame, np.uint8)
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f"[WARNING] Failed to decode frame from Robot ID: {req.robot_id}, Timestamp: {req.timestamp}")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("Invalid image data received.")
                    yield detection_pb2.DetectionResponse(
                        robot_id=req.robot_id,
                        timestamp=req.timestamp
                    )
                    continue

                print(f"[INFO] Frame shape: {frame.shape} from Robot ID: {req.robot_id}")

                # Perform detection
                detections = self.detector.detect(frame)
                print(f"[INFO] Detected {len(detections)} objects for Robot ID: {req.robot_id}")

                # Build response with metadata
                response = detection_pb2.DetectionResponse(
                    robot_id=req.robot_id,
                    timestamp=req.timestamp
                )

                for det in detections:
                    box = detection_pb2.DetectionBox(
                        label=det["label"],
                        confidence=det["confidence"],
                        x1=det["box"][0],
                        y1=det["box"][1],
                        x2=det["box"][2],
                        y2=det["box"][3]
                    )
                    response.boxes.append(box)

                print(f"[INFO] Sending detection response for Robot ID: {req.robot_id}")
                yield response

            except Exception as e:
                print(f"[ERROR] Exception during detection for Robot ID: {req.robot_id}: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Server error: {str(e)}")
                yield detection_pb2.DetectionResponse(
                    robot_id=req.robot_id,
                    timestamp=req.timestamp
                )


def serve(
    host: str = "[::]",
    port: int = 50051,
    max_workers: int = 10,
    model_path: str = "yolov9-c-fire.pt",
    use_gpu: bool = True
):
    """
    Launch the gRPC server for object detection.
    """
    print(f"[INFO] Starting gRPC server on {host}:{port} with {max_workers} threads...")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    detection_servicer = DetectionServicer(model_path=model_path, use_gpu=use_gpu)

    detection_pb2_grpc.add_DetectionServiceServicer_to_server(detection_servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print("[INFO] Server started and listening.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server gracefully...")
        server.stop(grace=None)


if __name__ == "__main__":
    # Optional: Use environment variables or CLI args for config
    serve(
        host=os.getenv("YOLO_SERVER_HOST", "[::]"),
        port=int(os.getenv("YOLO_SERVER_PORT", "50051")),
        max_workers=int(os.getenv("YOLO_MAX_WORKERS", "10")),
        model_path=os.getenv("YOLO_MODEL_PATH", "best.pt"),
        use_gpu=os.getenv("YOLO_USE_GPU", "1") == "1"
    )
