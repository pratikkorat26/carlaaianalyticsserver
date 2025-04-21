#!/usr/bin/env python
# streaming_server/client.py

"""
Real-time object detection and tracking client that connects to a gRPC
detection service, streams video frames, and displays annotated results.
"""

import os
import sys
import time
import logging
from typing import Dict, Iterator, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import cv2
import grpc
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tracking-client")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add proto paths
from protos import detection_pb2, detection_pb2_grpc

# Type definitions
FrameType = np.ndarray
ColorType = Tuple[int, int, int]


@dataclass
class ClientConfig:
    """Configuration parameters for the client."""
    server_address: str = "localhost:50051"
    camera_source: int = 0  # 0 for default webcam
    frame_skip: int = 2  # Process every n-th frame
    display_width: int = 1280
    display_height: int = 720
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds
    table_width: int = 500
    font_scale: float = 0.5

    @classmethod
    def from_env(cls) -> 'ClientConfig':
        """Load configuration from environment variables."""
        return cls(
            server_address=os.getenv("TRACKING_SERVER_ADDRESS", "localhost:50051"),
            camera_source=int(os.getenv("TRACKING_CAMERA_SOURCE", "0")),
            frame_skip=int(os.getenv("TRACKING_FRAME_SKIP", "2")),
            display_width=int(os.getenv("TRACKING_DISPLAY_WIDTH", "1280")),
            display_height=int(os.getenv("TRACKING_DISPLAY_HEIGHT", "720")),
            max_retries=int(os.getenv("TRACKING_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("TRACKING_RETRY_DELAY", "2.0")),
            table_width=int(os.getenv("TRACKING_TABLE_WIDTH", "500")),
            font_scale=float(os.getenv("TRACKING_FONT_SCALE", "0.5")),
        )


class ColorManager:
    """Manages consistent colors for track IDs."""

    def __init__(self):
        self._colors: Dict[int, ColorType] = {}

    def get_color(self, track_id: int) -> ColorType:
        """Get a consistent color for a track ID."""
        if track_id not in self._colors:
            self._colors[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        return self._colors[track_id]


class FrameProcessor:
    """Handles frame encoding and decoding operations."""

    @staticmethod
    def encode_frame(frame: FrameType, quality: int = 90) -> Optional[bytes]:
        """Encode a frame as JPEG bytes."""
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            return buffer.tobytes() if success else None
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            return None

    @staticmethod
    def decode_frame(frame_bytes: bytes) -> Optional[FrameType]:
        """Decode a frame from bytes."""
        try:
            frame_np = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            return None

    @staticmethod
    def resize_frame(frame: FrameType, width: int, height: int) -> FrameType:
        """Resize a frame to the specified dimensions."""
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class UIRenderer:
    """Renders user interface elements like tracking tables."""

    def __init__(self, color_manager: ColorManager, config: ClientConfig):
        self.color_manager = color_manager
        self.config = config

    def create_tracking_table(self, objects) -> FrameType:
        """Create an image displaying tracking data in a table format."""
        width = self.config.table_width
        font_scale = self.config.font_scale

        if not objects:
            blank = np.ones((100, width, 3), dtype=np.uint8) * 255
            cv2.putText(blank, "No objects tracked", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
            return blank

        # Column headers
        headers = ["ID", "Class", "Confidence", "Position"]

        # Prepare data rows
        rows = []
        for obj in objects:
            track_id = obj.track_id
            label = obj.label
            conf = f"{obj.confidence:.2f}"
            pos = f"({int(obj.x1)},{int(obj.y1)})-({int(obj.x2)},{int(obj.y2)})"
            rows.append([str(track_id), label, conf, pos])

        # Calculate dimensions
        line_height = 25
        height = (len(rows) + 2) * line_height  # +2 for header and padding

        # Create blank image
        table_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw column headers
        y = line_height
        col_widths = [50, 120, 100, 230]  # Adjust column widths as needed
        x_positions = [0]
        for i in range(len(col_widths)):
            x_positions.append(x_positions[i] + col_widths[i])

        # Draw header row
        for i, header in enumerate(headers):
            cv2.putText(table_img, header, (x_positions[i] + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

        # Draw header separator line
        y += 5
        cv2.line(table_img, (0, y), (width, y), (0, 0, 0), 1)
        y += 10

        # Draw data rows
        for row in rows:
            for i, cell in enumerate(row):
                # Get color for ID column
                color = (0, 0, 0)
                if i == 0:  # ID column
                    track_id = int(cell)
                    color = self.color_manager.get_color(track_id)

                cv2.putText(table_img, cell, (x_positions[i] + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

            y += line_height

            # Draw row separator
            cv2.line(table_img, (0, y - 10), (width, y - 10), (200, 200, 200), 1)

        return table_img


class TrackingClient:
    """gRPC client for object detection and tracking service."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.color_manager = ColorManager()
        self.frame_processor = FrameProcessor()
        self.ui_renderer = UIRenderer(self.color_manager, config)
        self.channel = None
        self.stub = None

    def connect(self) -> bool:
        """Connect to the gRPC server."""
        try:
            # Create secure channel with appropriate options
            self.channel = grpc.insecure_channel(
                self.config.server_address,
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
                    ('grpc.keepalive_time_ms', 30000),  # Send keepalive ping every 30s
                    ('grpc.keepalive_timeout_ms', 10000),  # Keepalive ping timeout after 10s
                ]
            )
            self.stub = detection_pb2_grpc.DetectionServiceStub(self.channel)
            logger.info(f"Connected to tracking server at {self.config.server_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    def close(self) -> None:
        """Close the channel to the gRPC server."""
        if self.channel:
            self.channel.close()
            logger.info("Connection to tracking server closed")

    def request_generator(self, cap: cv2.VideoCapture) -> Iterator[detection_pb2.FrameRequest]:
        """Generate streaming requests from a video capture device."""
        count = 0
        robot_id = os.getenv("ROBOT_ID", "robot_1")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            # Optional frame throttling
            if count % self.config.frame_skip != 0:
                count += 1
                continue

            # Resize frame if needed
            if frame.shape[1] != self.config.display_width or frame.shape[0] != self.config.display_height:
                frame = self.frame_processor.resize_frame(
                    frame, self.config.display_width, self.config.display_height
                )

            # Encode frame
            encoded = self.frame_processor.encode_frame(frame)
            if encoded:
                yield detection_pb2.FrameRequest(
                    robot_id=robot_id,
                    timestamp=int(time.time() * 1000),
                    frame=encoded
                )
            else:
                logger.warning("Failed to encode frame, skipping")

            count += 1

    def process_responses(self, response_stream) -> None:
        """Process streaming responses from the server."""
        window_name_video = "Object Tracking"
        window_name_table = "Tracking Data"

        # Create windows
        cv2.namedWindow(window_name_video, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name_table, cv2.WINDOW_NORMAL)

        try:
            for response in response_stream:
                if not response.annotated_frame:
                    logger.warning("Received empty response frame")
                    continue

                # Decode the annotated frame
                annotated_frame = self.frame_processor.decode_frame(response.annotated_frame)
                if annotated_frame is None:
                    logger.warning("Failed to decode annotated frame")
                    continue

                # Create tracking data table
                tracking_table = self.ui_renderer.create_tracking_table(response.objects)

                # Display the annotated frame and tracking table
                cv2.imshow(window_name_video, annotated_frame)
                cv2.imshow(window_name_table, tracking_table)

                # Print tracking info
                logger.debug(f"Received {len(response.objects)} tracked objects")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit")
                    break
        except grpc.RpcError as e:
            logger.error(f"RPC error during streaming: {e.code()}: {e.details()}")
        except Exception as e:
            logger.exception(f"Error processing responses: {e}")
        finally:
            cv2.destroyAllWindows()

    def run(self) -> None:
        """Run the client application."""
        retries = 0

        while retries < self.config.max_retries:
            try:
                # Initialize camera
                cap = cv2.VideoCapture(self.config.camera_source)
                if not cap.isOpened():
                    logger.error(f"Cannot open camera source: {self.config.camera_source}")
                    return

                # Set camera properties if needed
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.display_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.display_height)

                # Connect to server
                if not self.connect():
                    retries += 1
                    time.sleep(self.config.retry_delay)
                    continue

                # Stream frames and get responses
                response_stream = self.stub.StreamFrames(self.request_generator(cap))

                # Process responses in a separate thread
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.process_responses, response_stream)
                    future.result()  # Wait for processing to complete

                # Clean up
                cap.release()
                self.close()
                break  # Exit the retry loop on success

            except grpc.RpcError as e:
                logger.error(f"RPC error: {e.code()}: {e.details()}")
                retries += 1
                time.sleep(self.config.retry_delay)

            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                retries += 1
                time.sleep(self.config.retry_delay)

        if retries >= self.config.max_retries:
            logger.error(f"Failed after {self.config.max_retries} attempts")


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = ClientConfig.from_env()

        # Create and run client
        client = TrackingClient(config)
        client.run()

    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())