"""
Multi-Object Tracking Module for MotionVision TD
Implements Kalman-predictive ByteTrack / IoU association with persistent IDs and occlusion memory.
"""

import numpy as np

class Track:
    def __init__(self, track_id, bbox, cls_name, confidence, dominant_color):
        self.track_id = track_id
        self.bbox = list(bbox) # [x1, y1, x2, y2]
        self.cls_name = cls_name
        self.confidence = confidence
        self.dominant_color = dominant_color
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.velocity = [0.0, 0.0]
        self.history = [self.bbox]

    def update(self, bbox, confidence, dominant_color):
        # Compute smooth velocity
        cx_old = (self.bbox[0] + self.bbox[2]) / 2.0
        cy_old = (self.bbox[1] + self.bbox[3]) / 2.0
        cx_new = (bbox[0] + bbox[2]) / 2.0
        cy_new = (bbox[1] + bbox[3]) / 2.0
        vx = cx_new - cx_old
        vy = cy_new - cy_old

        alpha = 0.6
        self.velocity[0] = alpha * vx + (1 - alpha) * self.velocity[0]
        self.velocity[1] = alpha * vy + (1 - alpha) * self.velocity[1]

        self.bbox = list(bbox)
        self.confidence = confidence
        # Running average of appearance color
        self.dominant_color = [
            int(0.8 * self.dominant_color[i] + 0.2 * dominant_color[i]) for i in range(3)
        ]
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.bbox)

    def predict(self):
        # Predict next position using velocity
        x1, y1, x2, y2 = self.bbox
        pred_x1 = x1 + self.velocity[0]
        pred_y1 = y1 + self.velocity[1]
        pred_x2 = x2 + self.velocity[0]
        pred_y2 = y2 + self.velocity[1]
        self.bbox = [pred_x1, pred_y1, pred_x2, pred_y2]
        self.age += 1
        self.time_since_update += 1
        return self.bbox

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

class MultiObjectTracker:
    def __init__(self, max_lost_frames=45, iou_threshold=0.25):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        Takes detections: list of dicts with 'bbox', 'class_name', 'confidence', 'dominant_color'
        Returns: list of active Track objects
        """
        # Step 1: Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]
            return [t for t in self.tracks if t.time_since_update == 0]

        # Step 2: Match detections with existing tracks
        matched_tracks = set()
        matched_dets = set()

        if len(self.tracks) > 0 and len(detections) > 0:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    if t.cls_name == d['class_name']:
                        iou_val = compute_iou(t.bbox, d['bbox'])
                        # Color affinity bonus
                        cdist = np.linalg.norm(np.array(t.dominant_color) - np.array(d['dominant_color'])) / 441.67 # normalized [0,1]
                        score = iou_val * 0.8 + (1.0 - cdist) * 0.2
                        iou_matrix[i, j] = score

            # Greedy bipartite matching
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                row, col = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if row in matched_tracks or col in matched_dets:
                    iou_matrix[row, col] = 0.0
                    continue

                # Match found!
                matched_tracks.add(row)
                matched_dets.add(col)
                det = detections[col]
                self.tracks[row].update(det['bbox'], det['confidence'], det['dominant_color'])
                iou_matrix[row, :] = 0.0
                iou_matrix[:, col] = 0.0

        # Step 3: Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = Track(
                    track_id=self.next_id,
                    bbox=det['bbox'],
                    cls_name=det['class_name'],
                    confidence=det['confidence'],
                    dominant_color=det['dominant_color']
                )
                self.tracks.append(new_track)
                self.next_id += 1

        # Step 4: Remove expired tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost_frames]

        # Return tracks updated in current frame or recently predicted
        active_tracks = [t for t in self.tracks if t.time_since_update < 3]
        return active_tracks
