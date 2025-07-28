# report_generator.py
import os
import pandas as pd
from ultralytics import YOLO
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Update these paths
MODEL_PATH = "C:/Users/anvesha singh/runs/detect/sixray_exp2/weights/best.pt"
IMAGE_FOLDER = "C:/Users/anvesha singh/Documents/SIXray/test/images"
OUTPUT_CSV = "C:/Users/anvesha singh/Documents/SIXray/threat_report.csv"
# ---------------------

# Load the model
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# Initialize a counter for all detected classes
total_threat_counts = Counter()

# Get a list of all image files
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"Found {len(image_files)} images to process.")

# Process each image in the folder with a progress bar
for filename in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(IMAGE_FOLDER, filename)

    # Run prediction
    results = model(image_path, verbose=False) # verbose=False to reduce console output

    # Get detected class names for the first result
    detected_classes = results[0].boxes.cls
    class_names = [model.names[int(c)] for c in detected_classes]

    # Update the total counts
    total_threat_counts.update(class_names)

# Convert the counter to a pandas DataFrame
report_df = pd.DataFrame.from_dict(total_threat_counts, orient='index', columns=['Count'])
report_df.index.name = 'Threat_Class'
report_df = report_df.sort_values(by='Count', ascending=False)

# Save the DataFrame to a CSV file
report_df.to_csv(OUTPUT_CSV)

print("\n--- Threat Report ---")
print(report_df)
print(f"\n✅ Report successfully saved to {OUTPUT_CSV}")