# import cv2
# from ultralytics.models.sam import SAM2VideoPredictor

# def get_point_from_click(video_path):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise Exception(f"Unable to open video: {video_path}")
    
#     # Read the first frame
#     ret, frame = cap.read()
#     if not ret:
#         raise Exception("Unable to read the first frame from the video.")
    
#     # Display the first frame in a pop-up window
#     point = []

#     def mouse_callback(event, x, y, flags, param):
#         nonlocal point
#         if event == cv2.EVENT_LBUTTONDOWN:
#             point = [x, y]
#             print(f"Point selected: {point}")
#             cv2.destroyAllWindows()

#     cv2.imshow("Select a Point", frame)
#     cv2.setMouseCallback("Select a Point", mouse_callback)

#     # Wait for the user to select a point
#     while True:
#         if cv2.waitKey(1) & 0xFF == 27 or point:
#             break

#     cap.release()
#     return point

# # Video path
# video_path = "/mnt/data/demo_video/output.mp4"

# # Get the point selected by the user
# selected_point = [623, 1119]#get_point_from_click(video_path)

# if selected_point:
#     print(f"Selected point: {selected_point}")
    
#     # Create SAM2VideoPredictor
#     overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2.1_l.pt")
#     predictor = SAM2VideoPredictor(overrides=overrides, samurai=True)

#     # Run inference using the selected point
#     results = predictor(source=video_path, points=[selected_point], labels=[1])
#     print("Inference completed.")
# else:
#     print("No point was selected.")

import cv2
from ultralytics.models.sam import SAM2VideoPredictor

def get_point_from_click(frame, window_name="Select a Point"):
    """
    Opens a window with the provided frame and allows the user to click on the frame to select a point.
    """
    point = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = [x, y]
            print(f"Point selected: {point}")
            cv2.destroyAllWindows()

    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Wait for the user to select a point or press ESC
    while True:
        if cv2.waitKey(1) & 0xFF == 27 or point:
            break

    return point

def get_frame_at_time(video_path, time_in_seconds):
    """
    Returns the frame at the specified time in seconds from the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Unable to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_in_seconds)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception(f"Unable to read the frame at {time_in_seconds} seconds.")
    
    return frame

# Video path
video_path = "/mnt/data/demo_video/output.mp4"

# Get the point selected from the start frame (0 seconds)
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
cap.release()

if not ret:
    raise Exception("Unable to read the first frame from the video.")
    
print("Select the point from the first frame (0 seconds):")
# start_point = get_point_from_click(first_frame, "Select a Point (Start Frame)")

# Get the point selected from the 7-second frame
print("Select the point from the 7-second frame:")
# seven_second_frame = get_frame_at_time(video_path, 7)
# selected_point_7sec = get_point_from_click(seven_second_frame, "Select a Point (7 Seconds)")

# If both points are selected, proceed with the predictor
if True :#and selected_point_7sec
    # print(f"Selected points: Start Frame: {start_point}, 7-Second Frame: {selected_point_7sec}")
    
    # Create SAM2VideoPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2.1_l.pt")
    predictor = SAM2VideoPredictor(overrides=overrides, samurai=True)

    # Run inference using the selected points
    # results = predictor(source=video_path, points=[start_point, selected_point_7sec], labels=[1, 1])
    results = predictor(source=video_path, points=[[620, 1103]], labels=[1])
    print("Inference completed.")
else:
    print("No points were selected.")
