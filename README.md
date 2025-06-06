# Cup Tracking with YOLOv8 and DeepSORT

This project implements real-time cup/mug tracking in videos using YOLOv8 object detection and DeepSORT tracking algorithm. The system detects and tracks multiple cups/mugs in a video stream, assigns unique IDs to each cup, and visualizes their trajectories with different colors.

## Key Features
- Real-time cup detection using YOLOv8 nano model
- Multi-object tracking with enhanced DeepSORT (max_age=30, max_lost=5, iou_threshold=0.3)
- Unique ID and color assignment for each tracked cup
- Trajectory visualization (last 50 points)
- Confidence score display for each detection
- Additional object detection display (in gray)

## Requirements
- Python 3.x
- OpenCV (cv2)
- Ultralytics YOLO
- Custom EnhancedCupTracker implementation

## Installation

1. Install the required packages:
```bash
pip install ultralytics opencv-python
```

2. Ensure you have the YOLOv8 nano model file (`yolov8n.pt`) in your working directory.

## Usage

Run the tracking script with your video file:
```bash
python videotrack_new.py --input path/to/your/video.mp4 --output output.mp4
```

### Arguments
- `--input`: Path to input video file (default: input_video.mp4)
- `--output`: Path to output video file (default: output_detected.mp4)

## Features in Detail

### Detection
- Uses YOLOv8 nano model for efficient real-time detection
- Focused on cup detection (COCO class ID 41)
- Confidence threshold of 0.3 for reliable detections

### Tracking
- Enhanced DeepSORT implementation
- Maintains persistent IDs across video frames
- Handles occlusions and track management
- Maximum track age: 30 frames
- Maximum lost frames: 5
- IOU threshold: 0.3

### Visualization
- Unique color generation for each tracked cup
- Trajectory history limited to 50 points for efficiency
- Bounding boxes with:
  - Detection confidence scores
  - Track IDs
  - Center point markers
- Gray-scale visualization for other detected objects
