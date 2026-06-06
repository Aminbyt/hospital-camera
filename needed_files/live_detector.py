import cv2
from ultralytics import YOLO

# 1. Load YOUR custom AI brain
# (If you moved or renamed best.pt, just change this path to match!)
MODEL_PATH = 'runs/detect/train/weights/best.pt'
model = YOLO(MODEL_PATH)

# 2. Start the Webcam (Change to 1 if OBS is blocking it)
cap = cv2.VideoCapture(1)

print("🚀 Starting Live PPE Detection...")
print("Press 'q' on your keyboard to quit the video window.")

while True:
    # Read the live camera frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read from webcam.")
        break

    # 3. Let YOLO look at the frame
    # (conf=0.6 means it will only draw a box if it is 60% sure)
    results = model(frame, stream=True, conf=0.6)

    # 4. Draw the boxes and labels on the image
    for r in results:
        # r.plot() is YOLO's built-in magic function that draws the boxes for us!
        annotated_frame = r.plot()

    # 5. Show the video window on your screen
    cv2.imshow('Hospital PPE Monitor', annotated_frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when you quit
cap.release()
cv2.destroyAllWindows()