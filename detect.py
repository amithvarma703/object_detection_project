# detect.py - Starter object detection script using Ultralytics YOLO

from ultralytics import YOLO
import cv2

def main():
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # small model for demo
    
    # Run inference on sample images folder
    results = model.predict(source='sample_images', save=True)
    
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
