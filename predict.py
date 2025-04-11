import torch
from ultralytics import YOLO
import cv2

from PIL import Image


source_image = "/Users/akashmanna/Downloads/27522.jpg"
#"/Users/akashmanna/ws/opervu/imgaeStitching/ImageBelnding/blended_images/cam3/output/final_composite.jpg"
image = Image.open(source_image)





resized_image = image.resize((2048, 2048), Image.BILINEAR)

print(">>>>>>", resized_image.size)
model = YOLO("weights/best_wo_specialised_training.pt")
# Perform inference on an image
print(model.names)
results = model(source=resized_image, conf = 0.85, iou=0.25, imgsz=2480, save=True)