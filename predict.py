import torch
from ultralytics import YOLO

model = YOLO("/mnt/data/weights/base_weight/weights/best_wo_specialised_training.pt")
# Perform inference on an image
print(model.names)
results = model(source="/mnt/data/demo_video/video_2255.avi", conf = 0.25, iou=0.25, imgsz=2480, save=True)