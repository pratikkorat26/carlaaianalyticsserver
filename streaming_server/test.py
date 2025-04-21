# test_proto_check.py
import detection_pb2

# Instantiate a test FrameRequest
req = detection_pb2.FrameRequest(
    robot_id="robot_1",
    timestamp=1234567890,
    frame=b"\x00\x01\x02"
)

print("✅ FrameRequest:", req)

# Instantiate a response with tracked object
track = detection_pb2.TrackedObject(
    label="person",
    confidence=0.95,
    x1=100, y1=100, x2=200, y2=200,
    track_id=1
)

response = detection_pb2.AnnotatedResponse(
    robot_id="robot_1",
    timestamp=1234567890,
    annotated_frame=b"...jpeg..."
)
response.objects.append(track)

print("✅ AnnotatedResponse:", response)
