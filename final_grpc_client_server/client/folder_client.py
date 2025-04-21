import os
import cv2
import time
import grpc
import traceback
import numpy as np
from collections import deque
from random import randint
import sys

# Add parent directory to Python path to import detection_pb2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from protos import detection_pb2, detection_pb2_grpc

# --- Color cache ---
CLASS_COLORS = {}

def get_color(label: str):
    if label not in CLASS_COLORS:
        CLASS_COLORS[label] = (randint(0, 255), randint(0, 255), randint(0, 255))
    return CLASS_COLORS[label]

# --- Draw boxes ---
def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    try:
        for box in detections:
            x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
            label = f"{box.label} {box.confidence:.2f}"
            color = get_color(box.label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        print(f"⚠️ Error drawing detections: {e}")
    return frame

# --- Encode image to JPEG bytes ---
def encode_frame(frame: np.ndarray) -> bytes:
    try:
        success, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes() if success else None
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
        return None

# --- Request generator for folder streaming ---
def request_generator_from_folder(folder_path, frame_queue, filename_queue, stop_when_done=False):
    seen = set()
    empty_checks = 0
    while True:
        try:
            files = sorted(os.listdir(folder_path))
            new_images_found = False

            for fname in files:
                if fname.endswith(".jpg") and fname not in seen:
                    new_images_found = True
                    seen.add(fname)
                    path = os.path.join(folder_path, fname)

                    frame = cv2.imread(path)
                    if frame is None:
                        print(f"⚠️ Skipped corrupted image: {fname}")
                        continue

                    encoded = encode_frame(frame)
                    if encoded:
                        frame_queue.append(frame)
                        filename_queue.append(fname)
                        yield detection_pb2.FrameRequest(
                            robot_id="robot_1",
                            timestamp=int(time.time() * 1000),
                            frame=encoded
                        )

            if not new_images_found:
                empty_checks += 1
                if stop_when_done and empty_checks >= 5:  # checked 5 times in a row
                    print("✅ All images processed. Stopping stream.")
                    return
            else:
                empty_checks = 0

        except Exception as e:
            print(f"❌ Generator error: {e}")
            traceback.print_exc()
        time.sleep(0.2)


# --- Main client logic with fault tolerance ---
def run_grpc_folder_client(folder_path: str, save_path: str, max_retries=5):
    print(f"🚀 Streaming images from: {folder_path}")
    os.makedirs(save_path, exist_ok=True)

    attempt = 0
    while attempt < max_retries:
        try:
            frame_queue = deque(maxlen=30)
            filename_queue = deque(maxlen=30)

            channel = grpc.insecure_channel("localhost:50051")
            stub = detection_pb2_grpc.DetectionServiceStub(channel)
            response_stream = stub.StreamFrames(
                request_generator_from_folder(folder_path, frame_queue, filename_queue, stop_when_done=True)
            )

            for response in response_stream:
                try:
                    if frame_queue and filename_queue:
                        frame = frame_queue.popleft()
                        fname = filename_queue.popleft()

                        annotated = draw_detections(frame, response.boxes)

                        # Save annotated frame
                        save_name = os.path.join(save_path, fname)
                        cv2.imwrite(save_name, annotated)

                        # Show
                        cv2.imshow("Detection - Folder Stream", annotated)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("🛑 Quit signal received.")
                            return
                except Exception as e:
                    print(f"⚠️ Frame processing error: {e}")
                    traceback.print_exc()

        except grpc.RpcError as e:
            print(f"🔌 gRPC error: {e.code()} - {e.details()}")
            traceback.print_exc()
            attempt += 1
            print(f"🔁 Retrying gRPC connection... ({attempt}/{max_retries})")
            time.sleep(3)

        except KeyboardInterrupt:
            print("🛑 Interrupted by user.")
            break

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            traceback.print_exc()
            break

    print("❗ Exiting client.")

    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"⚠️ OpenCV window cleanup failed: {e}")

if __name__ == "__main__":
    input_folder = "../utils/carla_images"
    output_folder = "../utils/predicted_image"
    run_grpc_folder_client(input_folder, output_folder)
