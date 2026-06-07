"""Hand Washing Detection Module - Detects proper hand washing technique."""

import time
import math
import cv2
import config


class HandWashDetector:
    """Detects and tracks hand washing activity with bubble zone logic."""

    def __init__(self):
        """Initialize hand washing state variables."""
        self.reset_state()

    def reset_state(self):
        """Reset all hand washing tracking variables."""
        self.current_wash_time = 0
        self.last_hand_seen_time = 0
        self.is_washing = False
        self.last_valid_wash_time = 0
        self.prev_hand_pts = None
        self.scrub_anchor_pos = None
        self.last_touch_time = 0
        self.scrub_bubble_radius = 200

    def extract_hand_points(self, hand_landmarks, frame_w, frame_h):
        """Extract key landmark points from a hand.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            list: [(x1, y1), (x2, y2), (x3, y3)] - wrist, thumb, middle finger
        """
        hand0 = hand_landmarks
        return [
            (hand0.landmark[0].x * frame_w, hand0.landmark[0].y * frame_h),  # wrist
            (hand0.landmark[4].x * frame_w, hand0.landmark[4].y * frame_h),  # thumb
            (hand0.landmark[8].x * frame_w, hand0.landmark[8].y * frame_h)   # middle finger
        ]

    def get_wrist_distance(self, hand1_landmarks, hand2_landmarks, frame_w, frame_h):
        """Calculate distance between two hand wrists.
        
        Args:
            hand1_landmarks: First hand MediaPipe landmarks
            hand2_landmarks: Second hand MediaPipe landmarks
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            float: Distance in pixels
        """
        w1 = hand1_landmarks.landmark[0]
        w2 = hand2_landmarks.landmark[0]
        
        return math.hypot((w1.x - w2.x) * frame_w, (w1.y - w2.y) * frame_h)

    def calculate_hand_size(self, hand_landmarks, frame_w, frame_h):
        """Estimate hand size from wrist to middle finger.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            float: Hand size in pixels
        """
        h1_wrist = hand_landmarks.landmark[0]
        h1_mid = hand_landmarks.landmark[12]
        
        hand_size = math.hypot(
            (h1_wrist.x - h1_mid.x) * frame_w,
            (h1_wrist.y - h1_mid.y) * frame_h
        )
        
        return hand_size

    def set_bubble_zone(self, anchor_x, anchor_y, hand_size):
        """Set the scrubbing bubble zone around hands.
        
        Args:
            anchor_x: X coordinate of anchor point
            anchor_y: Y coordinate of anchor point
            hand_size: Size of hand for radius calculation
        """
        self.scrub_anchor_pos = (anchor_x, anchor_y)
        self.scrub_bubble_radius = max(hand_size * config.HAND_SIZE_MULTIPLIER, 
                                      config.MIN_BUBBLE_RADIUS)

    def is_hand_in_zone(self, hand_point, frame_w, frame_h):
        """Check if hand is within scrubbing bubble zone.
        
        Args:
            hand_point: (x, y) coordinates of hand point
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            bool: True if hand is in zone
        """
        if not self.scrub_anchor_pos:
            return False
        
        curr_x = hand_point.x * frame_w
        curr_y = hand_point.y * frame_h
        
        dist_to_anchor = math.hypot(
            curr_x - self.scrub_anchor_pos[0],
            curr_y - self.scrub_anchor_pos[1]
        )
        
        return dist_to_anchor < self.scrub_bubble_radius

    def detect_washing(self, hand_results, frame_w, frame_h, sink_y_start, ai_models):
        """Detect if hands are actively washing.
        
        Args:
            hand_results: Hand detection from AI models
            frame_w: Frame width
            frame_h: Frame height
            sink_y_start: Y coordinate of sink line
            ai_models: AI models instance for calculations
            
        Returns:
            dict: {
                'actively_washing': bool,
                'hands_count': int,
                'anchor_pos': tuple or None,
                'bubble_radius': float,
                'prev_hand_pts': list
            }
        """
        actively_washing = False
        hands_count = 0

        if not hand_results['detected']:
            self.prev_hand_pts = None
            return {
                'actively_washing': False,
                'hands_count': 0,
                'anchor_pos': self.scrub_anchor_pos,
                'bubble_radius': self.scrub_bubble_radius,
                'prev_hand_pts': self.prev_hand_pts
            }

        hand_data = hand_results['hand_results']
        hands_count = len(hand_data.multi_hand_landmarks)

        # Extract current hand points
        hand0 = hand_data.multi_hand_landmarks[0]
        current_hand_pts = self.extract_hand_points(hand0, frame_w, frame_h)

        # Calculate movement speed
        movement_speed = ai_models.calculate_hand_movement(current_hand_pts, self.prev_hand_pts)
        self.prev_hand_pts = current_hand_pts

        is_touching = False

        # Check for two-hand interlocking
        if hands_count == 2:
            wrist_distance = self.get_wrist_distance(
                hand_data.multi_hand_landmarks[0],
                hand_data.multi_hand_landmarks[1],
                frame_w, frame_h
            )

            if wrist_distance > config.WRIST_DISTANCE_THRESHOLD:
                box1 = ai_models.get_hand_bbox(hand_data.multi_hand_landmarks[0], frame_w, frame_h)
                box2 = ai_models.get_hand_bbox(hand_data.multi_hand_landmarks[1], frame_w, frame_h)

                if ai_models.bboxes_intersect(box1, box2):
                    anchor_x = (box1[0] + box1[2] + box2[0] + box2[2]) / 4
                    anchor_y = (box1[1] + box1[3] + box2[1] + box2[3]) / 4

                    if anchor_y > sink_y_start:
                        is_touching = True
                        hand_size = self.calculate_hand_size(hand_data.multi_hand_landmarks[0], frame_w, frame_h)
                        self.set_bubble_zone(anchor_x, anchor_y, hand_size)

        # Determine if actively washing
        time_since_last_touch = time.time() - self.last_touch_time
        is_moving = movement_speed > 2.0

        if is_touching and is_moving:
            actively_washing = True
            self.last_touch_time = time.time()
        elif hands_count >= 1 and is_moving and time_since_last_touch < config.TOUCH_TIMEOUT:
            if self.is_hand_in_zone(hand0.landmark[9], frame_w, frame_h):
                actively_washing = True

        # Update timing
        current_time = time.time()
        if actively_washing:
            self.last_valid_wash_time = current_time
        else:
            if hands_count < 2 and (current_time - self.last_valid_wash_time) < 3.0:
                actively_washing = True

        return {
            'actively_washing': actively_washing,
            'hands_count': hands_count,
            'anchor_pos': self.scrub_anchor_pos,
            'bubble_radius': self.scrub_bubble_radius,
            'prev_hand_pts': self.prev_hand_pts,
            'hand0': hand0 if hands_count > 0 else None
        }

    def update_wash_time(self, actively_washing):
        """Update cumulative wash time.
        
        Args:
            actively_washing: bool - Whether hands are currently washing
        """
        if actively_washing:
            if self.is_washing:
                time_spent = time.time() - self.last_hand_seen_time
                self.current_wash_time += time_spent
            self.is_washing = True
            self.last_hand_seen_time = time.time()
        else:
            self.is_washing = False

    def get_wash_status(self, hands_count):
        """Get human-readable wash status message.
        
        Args:
            hands_count: Number of hands detected
            
        Returns:
            str: Status message
        """
        if self.current_wash_time >= config.MAX_WASH_TIME:
            return "MAXIMUM WASH REACHED: ✅"
        
        if self.current_wash_time >= config.MIN_WASH_TIME:
            base_text = "WASHING (MINIMUM REACHED ✅ - YOU CAN CONTINUE)"
        else:
            base_text = "WASHING IN PROGRESS ...⏳"

        if hands_count == 0:
            if self.current_wash_time >= config.MIN_WASH_TIME:
                return "WASH COMPLETE: ✅"
            elif self.current_wash_time > 0:
                return f"{base_text}\n[ PAUSED: RETURN HANDS TO ZONE ]"
            else:
                return "WASH TIMER:⏯️ Pause"
        elif hands_count == 1:
            return f"{base_text}\n[ ⚠️WARNING: USE BOTH HANDS ]"
        elif hands_count == 2:
            return f"{base_text}\n[⚠️ WARNING: INTERLOCK HANDS ]"

        return base_text

    def draw_bubble_zone(self, frame):
        """Draw the scrubbing bubble zone on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            frame: Frame with drawn bubble
        """
        if self.scrub_anchor_pos:
            radius = int(self.scrub_bubble_radius)
            cv2.circle(frame, (int(self.scrub_anchor_pos[0]), int(self.scrub_anchor_pos[1])), 
                      radius, (0, 255, 255), 2)
        
        return frame
