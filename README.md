# YOLOv11 gRPC Object Detection Service for SSRCP

* This project provides a gRPC-based inference server for running real-time object detection using the YOLOv11 model. The server accepts image frames via gRPC, performs detection, and streams results back to the client.
* Author: Pratik Korat
* Date: 04/20/2025
* SJSU ID: 017512508
* Course: CMPE 281 - Intelligent Cloud Computing
* Instructor: Dr. Jerry Gao

## Features

- 🚀 Fast object detection using YOLOv11.
- 🔗 gRPC streaming API for efficient real-time inference.
- 📦 Python server and client implementation.
- 📝 Easily extendable for video, image, and batch processing.

## Files Structure
```
- custom_model
    - runs  /custom training runs
    - yolo11n.pt  /yolov11 small model
    - yolo11s.pt  /yolov11 nano model config, both models are not uploaded here because of size

- final_grpc_client_server   / this code does not support object tracking
    - client                 / client code
    - server                 / server code
    - proto                  / proto file for grpc
    - runs                   / custom detection video

- streaming_server           / this code supports object tracking
    - client                 / client code
    - server                 / server code
    - proto                  / proto file for grpc

```

## Requirements
    - Python 3.12
    - ultralytics
    - opencv-python
    - grpcio
    - grpcio-tools
    - numpy
    - protobuf
    - torch
    - torchvision
    - Pillow
    - pycocotools

## Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/pratikkorat26/carlaaianalyticsserver
    cd final_grpc_client_server or streaming_server
    ```
2. **Protobuf files:**
    ```
    protobuf files are generated using the command
    ```
   
3. **Run Server:**
    - Object Detection Server without Object Tracking:
    ```
    cd final_grpc_client_server/server
    python server.py    # this will start the server and listen on port 50051
    ```
    - Object Detection Server with Object Tracking:
    ```
    cd streaming_server/server
    python server.py    # this will start the server and listen on port 50051
    ```
   
4. **Run Client:**
    - Object Detection Client without Object Tracking:
    ```
    cd final_grpc_client_server/client
    python client.py    # this will start the client and send image frames to the server
    ```
    - Object Detection Client with Object Tracking:
    ```
    cd streaming_server/client
    python client.py    # this will start the client and send image frames to the server
    ```
   
## Contact Information
For any questions or issues, please contact:
- **Pratik Korat**
- pratikkorat1@gmail.com

