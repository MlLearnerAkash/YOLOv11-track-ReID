from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/mnt/data/weights/all_data_v11/yolov11_all_data_02122024_5/weights/best.pt")
print(model.names)
# Define path to video file
source = "/mnt/data/demo_video/surg1_needle_1.avi"

# Run inference on the source
# results = model(source, stream=True)  # generator of Results objects

# for result in results:
#     print(result)


model.predict(source, show = True, save = True, imgsz = 2048)

#Generator:
results = model(source, stream=True)  # generator of Results objects

for result in results:
    print(result[0].boxes)