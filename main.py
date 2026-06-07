import cv2
import time
import mediapipe as mp
from ultralytics import YOLO

# 1. Load YOLOv8 (Offline PPE AI)
MODEL_PATH = 'runs/detect/train/weights/best.pt'
model = YOLO(MODEL_PATH)

# 2. Load MediaPipe (Offline Hand Tracking AI)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# 3. Setup Camera and Wash Timer Variables
cap = cv2.VideoCapture(1)                   
REQUIRED_WASH_TIME = 20  
current_wash_time = 0
last_hand_seen_time = 0
is_washing = False

print("🏥 Hospital Master System Started (100% Offline)")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_width = frame.shape[1]

    # ---------------------------------------------------------
    # AI SCAN 1: YOLO PPE DETECTION
    # ---------------------------------------------------------
    results = model(frame, stream=True, conf=0.6)
    has_mask, has_hat = False, False

    for r in results:
        frame = r.plot()  # Draw YOLO boxes
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            if class_name == 'mask':
                has_mask = True
            elif class_name == 'hat':
                has_hat = True

    # ---------------------------------------------------------
    # AI SCAN 2: MEDIAPIPE HAND TRACKING
    # ---------------------------------------------------------
    # MediaPipe requires RGB colors to work
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    hands_detected = False

    # If the AI sees hands, draw the skeleton and mark as True
    if hand_results.multi_hand_landmarks:
        hands_detected = True
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---------------------------------------------------------
    # MASTER WORKFLOW LOGIC & USER INTERFACE
    # ---------------------------------------------------------
    ui_color = (0, 0, 255)  # Default Red
    ui_text = ""

    # Phase 1: Checking PPE
    if not (has_mask and has_hat):
        ui_text = "STEP 1: Please put on Mask and Hat"
        ui_color = (0, 140, 255)  # Orange
        current_wash_time = 0  # Reset timer if they take PPE off!

    # Phase 2: PPE is good, start the Hand Wash Check
    elif current_wash_time < REQUIRED_WASH_TIME:
        if hands_detected:
            # Hands are on screen, calculate time spent washing
            if is_washing:
                time_spent = time.time() - last_hand_seen_time
                current_wash_time += time_spent

            is_washing = True
            last_hand_seen_time = time.time()

            time_left = int(REQUIRED_WASH_TIME - current_wash_time)
            ui_text = f"STEP 2: Washing... Keep Scrubbing! ({time_left}s left)"
            ui_color = (255, 255, 0)  # Cyan/Blue for water

        else:
            # They stopped washing!
            is_washing = False
            time_left = int(REQUIRED_WASH_TIME - current_wash_time)
            ui_text = f"PAUSED: Put hands in sink! ({time_left}s left)"
            ui_color = (0, 0, 255)  # Red warning

    # Phase 3: Done!
    else:
        ui_text = "✅ SUCCESS: PPE Compliant & Hands Washed!"
        ui_color = (0, 200, 0)  # Bright Green

    # Draw the UI Banner
    cv2.rectangle(frame, (0, 0), (frame_width, 60), ui_color, -1)
    cv2.putText(frame, ui_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Hospital Sink - Master System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()