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

    def is_scrubbing_forearm(self, scrubber_hand, arm_hand, frame_w, frame_h):
        """Mathematically checks if one hand is scrubbing the other arm's forearm/elbow."""
        # Arm vector (from middle finger base to wrist)
        dx = (arm_hand.landmark[0].x - arm_hand.landmark[9].x) * frame_w
        dy = (arm_hand.landmark[0].y - arm_hand.landmark[9].y) * frame_h

        # The forearm extends backward from the wrist
        wrist_x = arm_hand.landmark[0].x * frame_w
        wrist_y = arm_hand.landmark[0].y * frame_h
       
        # Approximate elbow position (about 2.5x the hand length down the arm)
        elbow_x = wrist_x + (dx * 2.5)
        elbow_y = wrist_y + (dy * 2.5)

        # Scrubber hand center
        scrub_x = scrubber_hand.landmark[9].x * frame_w
        scrub_y = scrubber_hand.landmark[9].y * frame_h

        # Calculate Point-to-Line Segment Distance
        l2 = (elbow_x - wrist_x)**2 + (elbow_y - wrist_y)**2
        if l2 == 0:
            return False
           
        t = max(0, min(1, ((scrub_x - wrist_x) * (elbow_x - wrist_x) + (scrub_y - wrist_y) * (elbow_y - wrist_y)) / l2))
        proj_x = wrist_x + t * (elbow_x - wrist_x)
        proj_y = wrist_y + t * (elbow_y - wrist_y)

        dist = math.hypot(scrub_x - proj_x, scrub_y - proj_y)

        # If the scrubber is within 1.5x hand size of the forearm line, it's washing the arm!
        arm_hand_size = math.hypot(dx, dy)
        return dist < (arm_hand_size * 1.5)

    def detect_washing(self, hand_results, frame_w, frame_h, sink_y_start, ai_models):
        current_time = time.time()
        actively_washing = False
        valid_wash_this_frame = False
        hands_count = 0
       
        self.scrub_anchor_pos = None

        # --- MOVEMENT CHECK ---
        if hand_results['detected']:
            hand_data = hand_results['hand_results']
            hands_count = len(hand_data.multi_hand_landmarks)
           
            max_movement_speed = 0
            current_all_pts = []
           
            for i in range(hands_count):
                hand = hand_data.multi_hand_landmarks[i]
                pts = self.extract_hand_points(hand, frame_w, frame_h)
                current_all_pts.append(pts)
               
                if self.prev_hand_pts and i < len(self.prev_hand_pts):
                    speeds = [math.hypot(c[0] - p[0], c[1] - p[1]) for c, p in zip(pts, self.prev_hand_pts[i])]
                    if speeds:
                        max_movement_speed = max(max_movement_speed, max(speeds))
           
            self.prev_hand_pts = current_all_pts
           
            if max_movement_speed > 1.5:
                self.last_move_time = current_time
        else:
            self.prev_hand_pts = None

        is_moving = (current_time - self.last_move_time) < 0.5

        # --- STRICT VALID WASH EVALUATION ---
        if hands_count == 2:
            hand1 = hand_data.multi_hand_landmarks[0]
            hand2 = hand_data.multi_hand_landmarks[1]
           
            y1 = hand1.landmark[9].y * frame_h
            y2 = hand2.landmark[9].y * frame_h
            both_in_sink = (y1 > sink_y_start) and (y2 > sink_y_start)

            box1 = ai_models.get_hand_bbox(hand1, frame_w, frame_h)
            box2 = ai_models.get_hand_bbox(hand2, frame_w, frame_h)
           
            # Rule 1: Hands are rubbing together
            intersecting = ai_models.bboxes_intersect(box1, box2)
           
            # Rule 2: One hand is rubbing the other arm's wrist/elbow
            forearm_wash = (self.is_scrubbing_forearm(hand1, hand2, frame_w, frame_h) or
                            self.is_scrubbing_forearm(hand2, hand1, frame_w, frame_h))

            if both_in_sink and is_moving and (intersecting or forearm_wash):
                valid_wash_this_frame = True
                self.scrub_anchor_pos = (int((box1[0]+box1[2]+box2[0]+box2[2])/4), int((box1[1]+box1[3]+box2[1]+box2[3])/4))
                self.scrub_bubble_radius = max(self.calculate_hand_size(hand1, frame_w, frame_h) * 2.0, 150)
               
        # NOTE: 1-Hand Logic is entirely removed. 1 Hand = False.

        # --- TIMERS & GRACE PERIOD UPDATES ---
        if valid_wash_this_frame:
            actively_washing = True
            self.last_valid_wash_time = current_time
        else:
            # Slashed grace period to 1.0s to strictly stop timer on hand separation
            time_since_valid = current_time - self.last_valid_wash_time
            if time_since_valid <= 1.0 and self.current_wash_time > 0:
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
                return f"{base_text}\n[⚠️ WARNING: RUB HANDS OR ARMS TOGETHER ]"

        return base_text

    def draw_bubble_zone(self, frame):
        if self.scrub_anchor_pos:
            cv2.circle(frame, self.scrub_anchor_pos, int(self.scrub_bubble_radius), (0, 255, 255), 2)
        return frame