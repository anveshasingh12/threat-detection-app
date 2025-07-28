from ultralytics import YOLO

# 🔸 Load the trained model
model = YOLO("C:/Users/anvesha singh/runs/detect/sixray_exp2/weights/best.pt")

# 🔸 Run prediction on the entire test image folder
model.predict(source="C:/Users/anvesha singh/Documents/SIXray/test/images", save=True)




