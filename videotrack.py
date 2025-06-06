import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import argparse

# Input and output video paths
parser = argparse.ArgumentParser(description='Cup tracking in video using YOLOv8 and DeepSORT')
parser.add_argument('--input', type=str, default='input_video.mp4', help='Path to input video file')
parser.add_argument('--output', type=str, default='output_detected.mp4', help='Path to output video file')
args = parser.parse_args()

INPUT_VIDEO_PATH = args.input
OUTPUT_VIDEO_PATH = args.output

# Load video file instead of camera
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Initialize YOLOv11n model and DeepSORT
model = YOLO("yolo11n.pt")
deepsort = DeepSort(max_age=30)

# COCO class ID for cup/mug detection (coffee mug is class 41)
CUP_CLASS_IDS = {41}  # cup only

# Store colors and trajectories
cup_trajectories = {}
cup_colors = {}

def generate_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def draw_detection(frame, x1, y1, x2, y2, conf, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f'Cup: {conf:.2f}'
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
    return (center_x, center_y)

# Get video properties for saving output
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = model(frame)[0]
        boxes = results.boxes
        detections = []
        other_objects = []

        if boxes is not None:
            for i in range(len(boxes.cls)):
                cls_id = int(boxes.cls[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                conf = float(boxes.conf[i])

                if cls_id in CUP_CLASS_IDS and conf > 0.3:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'cup'))
                    center = draw_detection(frame, x1, y1, x2, y2, conf)
                else:
                    other_objects.append((x1, y1, x2, y2, cls_id))

        tracks = deepsort.update_tracks(detections, frame=frame)

        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            active_ids.add(track_id)

            if track_id not in cup_colors:
                cup_colors[track_id] = generate_color()
                cup_trajectories[track_id] = []

            cup_trajectories[track_id].append(center)
            if len(cup_trajectories[track_id]) > 50:
                cup_trajectories[track_id].pop(0)

            points = cup_trajectories[track_id]
            color = cup_colors[track_id]
            for j in range(1, len(points)):
                cv2.line(frame, points[j - 1], points[j], color, 2)
            # Draw the track ID above the mug
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw other objects
        for x1, y1, x2, y2, cls_id in other_objects:
            label = model.names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Write the frame to output video
        out.write(frame)

    except Exception as e:
        print(f"Error: {str(e)}")
        continue

# Release resources
cap.release()
out.release()
print(f"Video saved to {OUTPUT_VIDEO_PATH}")
