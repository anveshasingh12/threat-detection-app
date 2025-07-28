from ultralytics import YOLO

# Load trained model using full path
model = YOLO('C:/Users/anvesha singh/runs/detect/sixray_exp2/weights/best.pt')

# Evaluate on test set
metrics = model.val(data='data/sixray.yaml', split='test')

