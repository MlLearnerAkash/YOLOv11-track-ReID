import torch
from ultralytics import YOLO

model = YOLO("/mnt/data/weights/all_data_v11/yolov11_all_data_02122024_5/weights/best.pt")
# Perform inference on an image
results = model(source="/mnt/data/demo_video/surg1_needle_1.avi", conf = 0.05, classes=[14], iou=0.25, imgsz=2496, save=True)