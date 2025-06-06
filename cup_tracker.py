import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class EnhancedCupTracker:
    def __init__(self, max_age=30, max_lost=5, iou_threshold=0.3):
        self.deepsort = DeepSort(max_age=max_age)
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.lost_tracks = {}  # Store recently lost tracks for recovery
        self.kalman_filters = {}  # KF for each track
        
    def _initialize_kalman(self, track_id, measurement):
        """Initialize Kalman Filter for a new track"""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(2) * 5
        
        # Process noise
        kf.Q = np.eye(4) * 0.1
        
        # Initial state
        kf.x = np.array([[measurement[0], measurement[1], 0, 0]]).T
        
        return kf
    
    def _predict_location(self, track_id):
        """Predict next location using Kalman Filter"""
        if track_id in self.kalman_filters:
            kf = self.kalman_filters[track_id]
            kf.predict()
            return kf.x[:2].flatten()
        return None
    
    def _update_kalman(self, track_id, measurement):
        """Update Kalman Filter with new measurement"""
        if track_id in self.kalman_filters:
            kf = self.kalman_filters[track_id]
            kf.update(measurement)
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to x1y1x2y2 format
        box1 = np.array([x1, y1, x1 + w1, y1 + h1])
        box2 = np.array([x2, y2, x2 + w2, y2 + h2])
        
        # Calculate intersection
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _attempt_track_recovery(self, detections):
        """Attempt to recover lost tracks based on predictions and IoU"""
        recovered_tracks = []
        
        for track_id, lost_data in list(self.lost_tracks.items()):
            predicted_pos = self._predict_location(track_id)
            if predicted_pos is None:
                continue
                
            # Create a box from predicted position (assuming similar size as last known)
            last_size = lost_data['size']
            predicted_box = [
                predicted_pos[0] - last_size[0]/2,
                predicted_pos[1] - last_size[1]/2,
                last_size[0],
                last_size[1]
            ]
            
            # Find best matching detection
            best_iou = 0
            best_detection = None
            
            for detection in detections:
                iou = self._calculate_iou(predicted_box, detection[0])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection = detection
            
            if best_detection is not None:
                recovered_tracks.append((track_id, best_detection))
                del self.lost_tracks[track_id]
                detections.remove(best_detection)
        
        return recovered_tracks, detections
    
    def update(self, detections, frame):
        """Update tracks with new detections, including recovery and smoothing"""
        # Try to recover lost tracks first
        recovered_tracks, remaining_detections = self._attempt_track_recovery(detections)
        
        # Add recovered tracks to detections with high confidence
        for track_id, detection in recovered_tracks:
            remaining_detections.append((detection[0], 0.9, 'cup'))  # High confidence for recovered tracks
        
        # Update DeepSORT with remaining detections
        tracks = self.deepsort.update_tracks(remaining_detections, frame=frame)
        
        # Update Kalman Filters and track lost objects
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            bbox = track.to_ltrb()
            center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
            
            # Initialize or update Kalman Filter
            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = self._initialize_kalman(track_id, center)
            else:
                self._update_kalman(track_id, center)
            
            # Remove from lost tracks if found
            if track_id in self.lost_tracks:
                del self.lost_tracks[track_id]
        
        # Update lost tracks
        active_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
        for track_id in list(self.kalman_filters.keys()):
            if track_id not in active_track_ids and track_id not in self.lost_tracks:
                # Store information about lost track
                bbox = next((track.to_ltrb() for track in tracks if track.track_id == track_id), None)
                if bbox is not None:
                    self.lost_tracks[track_id] = {
                        'frames_lost': 0,
                        'last_bbox': bbox,
                        'size': (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    }
        
        # Remove old lost tracks
        for track_id in list(self.lost_tracks.keys()):
            self.lost_tracks[track_id]['frames_lost'] += 1
            if self.lost_tracks[track_id]['frames_lost'] > self.max_lost:
                del self.lost_tracks[track_id]
                del self.kalman_filters[track_id]
        
        return tracks
