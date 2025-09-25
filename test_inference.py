import torch
from pathlib import Path
from PIL import Image
import cv2

# Path to your trained model weights (update if needed)
MODEL_PATH = "runs/train/exp/weights/best.pt"

# Directory containing your test images
TEST_DIR = Path("test_images")

# Output directory
OUTPUT_DIR = Path("runs/test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_inference():
    # Load model (YOLOv5/YOLOv8 style)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)

    # Iterate through test images
    for img_path in TEST_DIR.glob("*.*"):
        print(f"Processing {img_path}...")

        # Run inference
        results = model(str(img_path))

        # Save results with bounding boxes
        save_path = OUTPUT_DIR / img_path.name
        results.save(save_dir=OUTPUT_DIR)

        # Also show detections in terminal
        print(results.pandas().xyxy[0])  # bounding boxes dataframe

if __name__ == "__main__":
    run_inference()
