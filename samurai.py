from ultralytics import YOLO

# model = YOLO("/mnt/data/weights/all_data_v11/yolov11_all_data_02122024_5/weights/best.pt")

# results = model.predict(source="/mnt/data/demo_video/utput.mp4", conf=0.5, classes=[1] , device="0", imgsz=1024)

from ultralytics.models.sam import SAM2VideoPredictor

# Create SAM2VideoPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2.1_b.pt")
predictor = SAM2VideoPredictor(overrides=overrides, samurai=True)

# Run inference with single point
# results = predictor(source="/mnt/data/demo_video/surg_sponge.avi", points=[[750, 1040], [750, 1740], [1490, 1740], [1490, 1040]], labels=1)


# yolo_points = []
# for result in results:
#     for box in result.boxes:
#         x_min, y_min, x_max, y_max = box.xyxy.cpu().numpy()[0]  # Convert to numpy array
#         center_x = int((x_min + x_max) / 2)
#         center_y = int((y_min + y_max) / 2)
#         yolo_points.append([int(x_min), int(y_min), int(x_max), int(y_max)])
#         print("YOLO points:", yolo_points)


'''


'''

# # Example: Assign labels (adjust logic as per your requirements)
# labels = [1] * len(yolo_points)  # Assuming all points belong to the same class
# print("YOLO points:", yolo_points)
# # Step 2: Run SAM2VideoPredictor with points from YOLO
# overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2.1_b.pt")
sam_predictor = SAM2VideoPredictor(overrides=overrides, samurai=True)



# Run inference with multiple points
# results = predictor(source="/mnt/data/demo_video/surg_sponge.avi", points=[[750, 1040], [1490, 1740]], labels=[1, 1])

# Run inference with multiple points prompt per object
results = predictor(source="/mnt/data/demo_video/utput.mp4", points=[[924, 600]], labels=[1])