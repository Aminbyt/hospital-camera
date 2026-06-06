from ultralytics import  YOLO

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    print("Starting YOYLOv8 Training ...")

    results = model.train(
        data = "dataset_final/data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 16,
        device = 0
    )

print("Training Complete! Check your 'runs' folder for your new model ")