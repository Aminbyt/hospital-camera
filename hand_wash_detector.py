"""Hand Wash Detector - Pure math, state tracking, and geometry module."""

import time
import cv2
import numpy as np
import config

class HandWashDetector:
    def __init__(self, sink_name="SINK_1"):
        self.sink_name = sink_name
        self.polygon = getattr(config, f"{sink_name}_POLYGON", None)
        
        # Timing state (Strictly using monotonic time)
        self.current_wash_time = 0.0
        self.last_update_time = time.monotonic()
        
        self.last_mask_seen_time = time.monotonic()
        self.last_hat_seen_time = time.monotonic()
        
        # WHO state
        self.completed_steps = set()

    def update_wash_time(self, actively_washing):
        """Calculates delta time and increments wash duration if washing."""
        now = time.monotonic()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Clamp dt to prevent massive jumps if the thread was suspended
        if dt > 1.0:
            dt = 0.0
            
        if actively_washing:
            self.current_wash_time += dt

    def update_ppe_state(self, has_mask, has_hat):
        """Updates the last seen monotonic timestamp for PPE."""
        now = time.monotonic()
        if has_mask:
            self.last_mask_seen_time = now
        if has_hat:
            self.last_hat_seen_time = now

    def reset_state(self):
        """Resets all metrics for a new session."""
        self.current_wash_time = 0.0
        self.completed_steps.clear()
        
        now = time.monotonic()
        self.last_update_time = now
        self.last_mask_seen_time = now
        self.last_hat_seen_time = now

    def get_wash_status(self, hand_count):
        """Returns the current textual status of the sink."""
        if hand_count == 0:
            return "STANDBY"
        elif hand_count == 1:
            return "PLEASE USE BOTH HANDS"
        elif self.current_wash_time >= config.MIN_WASH_TIME:
            return "WASH COMPLETE!"
        else:
            return "WASHING..."

    def check_zone(self, hand_results, frame_w, frame_h):
        """
        Geometrically checks if both hands are inside the configured polygon ROI.
        Returns a dict: {'actively_washing': bool, 'in_zone': bool}
        """
        if not hand_results or hand_results['count'] == 0:
            return {'actively_washing': False, 'in_zone': False}

        # If no polygon is configured, assume the whole screen is valid
        if not self.polygon:
            actively_washing = (hand_results['count'] >= 2)
            return {'actively_washing': actively_washing, 'in_zone': True}

        pts = np.array(self.polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        hands_in_zone = 0
        if hand_results['hand_results'] and hand_results['hand_results'].multi_hand_landmarks:
            for hand_landmarks in hand_results['hand_results'].multi_hand_landmarks:
                # Use the wrist (landmark 0) to determine hand position
                wrist_x = int(hand_landmarks.landmark[0].x * frame_w)
                wrist_y = int(hand_landmarks.landmark[0].y * frame_h)
                
                # Point polygon test: >= 0 means inside or on the edge
                if cv2.pointPolygonTest(pts, (wrist_x, wrist_y), False) >= 0:
                    hands_in_zone += 1

        in_zone = (hands_in_zone > 0)
        actively_washing = (hands_in_zone >= 2)
        
        return {'actively_washing': actively_washing, 'in_zone': in_zone}

    def draw_bubble_zone(self, frame):
        """Draws the transparent ROI overlay on the camera frame."""
        if not self.polygon:
            return frame
            
        overlay = frame.copy()
        pts = np.array(self.polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Determine color based on wash completion
        if self.current_wash_time >= config.MIN_WASH_TIME:
            color = (0, 255, 0)   # Green when complete
        elif self.current_wash_time > 0:
            color = (0, 165, 255) # Orange while washing
        else:
            color = (255, 0, 0)   # Blue on standby
            
        cv2.fillPoly(overlay, [pts], color)
        
        # Blend the overlay for a transparent bubble effect (alpha = 0.2)
        return cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    def detect_washing(self, hand_results, frame_w, frame_h, sink_y_start, ai_models):
        """RESTORED FROM MAIN BRANCH: Strict zone checking and hand intersection (touching)."""
        if not hand_results or hand_results['count'] < 2:
            return {'actively_washing': False, 'in_zone': False}

        landmarks = hand_results['hand_results'].multi_hand_landmarks
        
        # 1. Enforce that wrists are below the red Alcohol Scrub Zone line
        if sink_y_start is not None:
            wrists_in_zone = 0
            for hl in landmarks[:2]:
                if int(hl.landmark[0].y * frame_h) >= sink_y_start:
                    wrists_in_zone += 1
            if wrists_in_zone < 2:
                return {'actively_washing': False, 'in_zone': False}

        # 2. Enforce that the bounding boxes of the hands physically intersect (touching)
        box1 = ai_models.get_hand_bbox(landmarks[0], frame_w, frame_h)
        box2 = ai_models.get_hand_bbox(landmarks[1], frame_w, frame_h)
        
        # Expand the boxes slightly (by 30 pixels) to account for 3D depth forgiveness
        margin = 30
        b1 = [box1[0]-margin, box1[1]-margin, box1[2]+margin, box1[3]+margin]
        b2 = [box2[0]-margin, box2[1]-margin, box2[2]+margin, box2[3]+margin]
        
        actively_washing = ai_models.bboxes_intersect(b1, b2)
        
        return {'actively_washing': actively_washing, 'in_zone': True}