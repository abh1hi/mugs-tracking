# Cup Tracking with YOLOv8 and DeepSORT

This project implements real-time cup/mug tracking in videos using YOLOv8 and DeepSORT tracking algorithm.

## Setup and Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd mugs_tracking
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the tracking script with your video file:
```bash
python videotrack.py --input path/to/your/video.mp4 --output output.mp4
```

### Arguments
- `--input`: Path to input video file (default: input_video.mp4)
- `--output`: Path to output video file (default: output_detected.mp4)

## Model Information

This project uses:
- YOLOv8n model for object detection
- DeepSORT for object tracking

The model is specifically configured to track cups/mugs (COCO class ID 41) but can be modified to track other objects.

## Test Data

You can test the tracker with any video containing cups or mugs. Some suggestions for testing:
1. Record a video of cups being moved around on a table
2. Use a webcam to capture live cup movements
3. Download sample videos from public datasets

## Features

- Real-time cup detection and tracking
- Unique ID assignment for each tracked cup
- Trajectory visualization with different colors for each cup
- Confidence score display
- Detection of other objects in the scene (displayed in gray)
