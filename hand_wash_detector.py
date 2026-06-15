"""Hand Washing Detection Module - Detects proper hand washing technique."""

import time
import math
import cv2
import config

class HandWashDetector:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.current_wash_time = 0.0
        self.last_hand_seen_time = 0.0
        self.is_washing = False
        self.last_valid_wash_time = 0.0
        self.prev_hand_pts = None
        
        self.last_move_time = 0.0
        self.last_touch_time = 0.0
        
        self.scrub_anchor_pos = None
        self.scrub_bubble_radius = 200

        self.last_mask_seen_time = 0
        self.last_hat_seen_time = 0

    def extract_hand_points(self, hand_landmarks, frame_w, frame_h):
        hand0 = hand_landmarks
        return [
            (hand0.landmark[0].x * frame_w, hand0.landmark[0].y * frame_h),
            (hand0.landmark[4].x * frame_w, hand0.landmark[4].y * frame_h),
            (hand0.landmark[8].x * frame_w, hand0.landmark[8].y * frame_h)
        ]

    def calculate_hand_size(self, hand_landmarks, frame_w, frame_h):
        h1_wrist = hand_landmarks.landmark[0]
        h1_mid = hand_landmarks.landmark[12]
        return math.hypot((h1_wrist.x - h1_mid.x) * frame_w, (h1_wrist.y - h1_mid.y) * frame_h)

    def detect_washing(self, hand_results, frame_w, frame_h, sink_y_start, ai_models):
        current_time = time.time()
        actively_washing = False
        hands_count = 0
        
        # Define this safely at the top
        time_since_touch = current_time - self.last_touch_time
        self.scrub_anchor_pos = None 

        if hand_results['detected']:
            hand_data = hand_results['hand_results']
            hands_count = len(hand_data.multi_hand_landmarks)
            
            # --- BUG FIX #1: CHECK MOVEMENT ON ALL HANDS ---
            max_movement_speed = 0
            current_all_pts = []
            
            for i in range(hands_count):
                hand = hand_data.multi_hand_landmarks[i]
                pts = self.extract_hand_points(hand, frame_w, frame_h)
                current_all_pts.append(pts)
                
                # Check speed against previous frame
                if self.prev_hand_pts and i < len(self.prev_hand_pts):
                    speeds = [math.hypot(c[0] - p[0], c[1] - p[1]) for c, p in zip(pts, self.prev_hand_pts[i])]
                    if speeds:
                        max_movement_speed = max(max_movement_speed, max(speeds))
            
            self.prev_hand_pts = current_all_pts
            
            # If ANY hand is scrubbing fast, we are moving!
            if max_movement_speed > 1.5:
                self.last_move_time = current_time
        else:
            self.prev_hand_pts = None

        # 0.5s movement memory prevents the bar from flickering
        is_moving = (current_time - self.last_move_time) < 0.5

        if hands_count == 2:
            hand1 = hand_data.multi_hand_landmarks[0]
            hand2 = hand_data.multi_hand_landmarks[1]
            
            y1 = hand1.landmark[9].y * frame_h
            y2 = hand2.landmark[9].y * frame_h
            both_in_sink = (y1 > sink_y_start) and (y2 > sink_y_start)

            box1 = ai_models.get_hand_bbox(hand1, frame_w, frame_h)
            box2 = ai_models.get_hand_bbox(hand2, frame_w, frame_h)
            intersecting = ai_models.bboxes_intersect(box1, box2)

            if both_in_sink:
                if intersecting:
                    if is_moving:
                        actively_washing = True
                        self.last_touch_time = current_time  # Save memory
                        self.last_valid_wash_time = current_time
                        self.scrub_anchor_pos = (int((box1[0]+box1[2]+box2[0]+box2[2])/4), int((box1[1]+box1[3]+box2[1]+box2[3])/4))
                        self.scrub_bubble_radius = max(self.calculate_hand_size(hand1, frame_w, frame_h) * 2.0, 150)
                else:
                    # --- BUG FIX #2: KEEP ELBOW WASHING ALIVE INDEFINITELY ---
                    if is_moving and time_since_touch < 4.0 and self.current_wash_time > 0:
                        actively_washing = True
                        self.last_touch_time = current_time  # Refresh the grace period!
                        self.last_valid_wash_time = current_time
                        self.scrub_anchor_pos = (int((box1[0]+box2[0])/2), int((box1[1]+box2[1])/2))
                        self.scrub_bubble_radius = max(self.calculate_hand_size(hand1, frame_w, frame_h) * 3.5, 200)
                    else:
                        self.last_touch_time = 0  # Wipe memory if they stop scrubbing

        elif hands_count == 1:
            hand1 = hand_data.multi_hand_landmarks[0]
            y1 = hand1.landmark[9].y * frame_h
            
            if y1 > sink_y_start:
                # 1-Hand Elbow Logic
                if is_moving and time_since_touch < 4.0 and self.current_wash_time > 0:
                    actively_washing = True
                    self.last_touch_time = current_time  # Refresh the grace period!
                    self.last_valid_wash_time = current_time
                    self.scrub_anchor_pos = (int(hand1.landmark[9].x * frame_w), int(hand1.landmark[9].y * frame_h))
                    self.scrub_bubble_radius = max(self.calculate_hand_size(hand1, frame_w, frame_h) * 2.0, 150)
                else:
                    self.last_touch_time = 0

        # RINSING / 0-HANDS GRACE PERIOD (2.5 seconds)
        if not actively_washing and hands_count == 0:
            time_since_valid = current_time - self.last_valid_wash_time
            if time_since_valid < 2.5 and self.current_wash_time > 0:
                actively_washing = True

        return {'actively_washing': actively_washing, 'hands_count': hands_count}

    def update_wash_time(self, actively_washing):
        current_time = time.time()
        if actively_washing:
            if self.is_washing:
                time_spent = current_time - self.last_hand_seen_time
                self.current_wash_time += time_spent
            self.is_washing = True
            self.last_hand_seen_time = current_time
        else:
            self.is_washing = False

    def get_wash_status(self, hands_count): 
        # EXACT MATCH TO UI_007 TEXT
        if self.current_wash_time >= config.MAX_WASH_TIME:
            return "MAXIMUM WASH REACHED: ✅"
       
        if self.current_wash_time >= config.MIN_WASH_TIME:
            base_text = "WASHING (MINIMUM REACHED ✅ - YOU CAN CONTINUE)"
        else:
            base_text = "WASHING IN PROGRESS ...⏳"

        if self.is_washing:
            return base_text
        else:
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
        if self.scrub_anchor_pos:
            cv2.circle(frame, self.scrub_anchor_pos, int(self.scrub_bubble_radius), (0, 255, 255), 2)
        return frame