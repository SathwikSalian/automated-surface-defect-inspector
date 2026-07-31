# File: train.py
# IDE Execution: Run directly inside VS Code Terminal using: python train.py

from ultralytics import YOLO
import os

def start_training():
    # 1. Load pre-trained lightweight YOLOv8 model
    model = YOLO("yolov8n.pt")

    # 2. Path to your dataset configuration file
    yaml_path = os.path.join("dataset", "data.yaml")

    # 3. Train the model
    print("Starting Model Training...")
    results = model.train(
        data=yaml_path,
        epochs=20,          # Set to 20 epochs for quick prototype training
        imgsz=640,          # Standard image resolution
        batch=16,           # Reduce to 8 if your machine runs out of RAM/GPU memory
        name="defect_model" # Folder name where trained weights will be saved
    )
    print("Training Complete! Trained model saved under runs/detect/defect_model/weights/best.pt")

if __name__ == "__main__":
    start_training()