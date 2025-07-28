from ultralytics import YOLO

# Train model
model = YOLO('yolov8n.pt')  # Or use yolov8s.pt or yolov8m.pt for bigger models
model.train(data='data/sixray.yaml', epochs=50, imgsz=640, batch=16, name='sixray_exp')
