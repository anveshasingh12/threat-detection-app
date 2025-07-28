import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("C:/Users/anvesha singh/runs/detect/sixray_exp2/weights/best.pt")

# Paths
image_folder = "C:/Users/anvesha singh/Documents/SIXray/test/images"
output_folder = "C:/Users/anvesha singh/Documents/SIXray/test/heatmaps"
os.makedirs(output_folder, exist_ok=True)

# Loop through all images
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        results = model.predict(image_path)[0]

        heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            heatmap[y1:y2, x1:x2] += conf

        heatmap = np.clip(heatmap / np.max(heatmap), 0, 1)
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlayed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.6, heatmap_color, 0.4, 0)

        # Save output
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

        print(f"Saved: {output_path}")
