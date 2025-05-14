# import torch
# import cv2
# from ultralytics import YOLO
# from draw_utils import display_items_count
# from process_frame import process_frame
# def expand_bbox(bbox_xyxy, padding=100, image_width=None, image_height=None):
#     """Expands bounding box by specified padding on all sides"""
#     x_min, y_min, x_max, y_max = bbox_xyxy
#     x_min = max(0, x_min - padding)
#     y_min = max(0, y_min - padding)
#     x_max = x_max + padding
#     y_max = y_max + padding
    
#     if image_width and image_height:
#         x_max = min(image_width, x_max)
#         y_max = min(image_height, y_max)
    
#     return [x_min, y_min, x_max, y_max]

# import math
# from shapely.geometry import Polygon


# def get_rectangle_corners(x_min, y_min, x_max, y_max):
#     # Deriving four corners from the (x_min, y_min, x_max, y_max)
#     bottom_left = (x_min, y_min)
#     bottom_right = (x_max, y_min)
#     top_right = (x_max, y_max)
#     top_left = (x_min, y_max)

#     # Return the four corners
#     return (bottom_left, bottom_right, top_right, top_left)

# def distance(p1, p2):
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# def rectangle_area(points):
#     if len(points) != 4:
#         raise ValueError("Need exactly 4 points")

#     # Assume points are ordered (clockwise or counterclockwise)
#     side1 = distance(points[0], points[1])
#     side2 = distance(points[1], points[2])

#     return side1 * side2
# def is_indide(roi, obj):
#     status = False
#     sx_min, sy_min, sx_max, sy_max = roi
#     ox_min, oy_min, ox_max, oy_max = obj

#     p_special_boundary = get_rectangle_corners(sx_min, sy_min, sx_max, sy_max)
#     p_object_boundary = get_rectangle_corners(ox_min, oy_min, ox_max, oy_max)

#     obj_boundary_list = list(p_object_boundary)
#     obj_area = rectangle_area(obj_boundary_list)

#     polygon_special_boundary = Polygon(p_special_boundary)
#     polygon_object_boundary = Polygon(p_object_boundary)

#     intersection_area = polygon_object_boundary.intersection(polygon_special_boundary).area

#     ratio = intersection_area/obj_area
#     if ratio >= 1/3:
#         status =True
#     return status

# def process_video(input_path, output_path, model_path, target_class=0, conf=0.25, iou=0.5):
#     # Initialize video capture
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         print(f"Error opening video file {input_path}")
#         return

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     # Load YOLO model
#     model = YOLO(model_path)
#     item_names ={value: key for key, value in model.names.items()}
#     print("Model class names:", model.names)




#     max_items_count = {key: 0 for key in item_names.keys()}
#     current_candidate_max = {key: 0 for key in item_names.keys()}  # Potential new maximum being verified
#     consecutive_frames_count = {key: 0 for key in item_names.keys()}  # Counter for consecutive frames
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform inference
#         results = model.predict(frame, conf=conf, iou=iou, imgsz=2080, verbose=False)
#         # Process detections
#         items_count = {key:0 for key,_ in item_names.items()}
#         for result in results:
#             for box in result.boxes:
#                 if int(box.cls.item()) == target_class:
#                     # Get and expand bounding box
#                     bbox_xyxy = box.xyxy.cpu().numpy().squeeze().tolist()
#                     adjusted_bbox = expand_bbox(bbox_xyxy, padding=100, 
#                                               image_width=frame_width, 
#                                               image_height=frame_height)
                    
#                     # Draw rectangle on frame
#                     x_min, y_min, x_max, y_max = map(int, adjusted_bbox)
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
#                                 (0, 255, 0), 2)
            
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze())
                
#                 if is_indide(map(int, adjusted_bbox), map(int, box.xyxy.cpu().numpy().squeeze())):

#                     # Draw detection box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), 
#                                 (0,0,255))
#                     items_count[model.names[box.cls.item()]] +=1

#         # Update maximum tracking logic
#         for key in item_names.keys():
#             current_count = items_count.get(key, 0)
            
#             if current_candidate_max[key] > max_items_count[key]:
#                 # We have an active candidate being verified
#                 if current_count >= current_candidate_max[key]:
#                     consecutive_frames_count[key] += 1
#                     if consecutive_frames_count[key] >= 4:
#                         # Update confirmed maximum
#                         max_items_count[key] = current_candidate_max[key]
#                         # Reset tracking variables
#                         current_candidate_max[key] = 0
#                         consecutive_frames_count[key] = 0
#                 else:
#                     # Candidate not maintained, reset
#                     current_candidate_max[key] = 0
#                     consecutive_frames_count[key] = 0
#             else:
#                 # Check for new potential maximum
#                 if current_count > max_items_count[key]:
#                     current_candidate_max[key] = current_count
#                     consecutive_frames_count[key] = 1


#         frame = display_items_count(frame, max_items_count)

#         # Write processed frame
#         out.write(frame)

#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Processing complete. Output saved to {output_path}")

# if __name__ == "__main__":
#     model_path = "/data/dataset/weights/base_weight/weights/best_wo_specialised_training.pt"
#     input_video = "/data/dataset/demo_video/final_210225/surg_sponge.avi"
#     output_video = "./video_processed.mp4"
    
#     process_video(
#         input_path=input_video,
#         output_path=output_video,
#         model_path=model_path,
#         target_class=0,
#         conf=0.25,
#         iou=0.5
#     )



###V2.0
import torch
import cv2
from ultralytics import YOLO
from draw_utils import display_items_count
from process_frame import process_frame

from ultralytics import solutions

from collections import defaultdict

import math
from shapely.geometry import Polygon

def expand_bbox(bbox_xyxy, padding=100, image_width=None, image_height=None):
    """Expands bounding box by specified padding on all sides"""
    x_min, y_min, x_max, y_max = bbox_xyxy
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = x_max + padding
    y_max = y_max + padding
    
    if image_width and image_height:
        x_max = min(image_width, x_max)
        y_max = min(image_height, y_max)
    
    return [x_min, y_min, x_max, y_max]

def classify_track(rect, points, per_segment=False):
    """
    Classify how a list of points moves relative to a rectangular ROI.

    Args:
        rect (tuple): (x1, y1, x2, y2) — two opposite corners of the rectangle.
        points (list of tuple): e.g. [(x0,y0,_), (x1,y1,_), ...]. Only x,y are used.
        per_segment (bool): if True, also return a list of classifications for each adjacent segment.

    Returns:
        overall (str): one of
            - "inside_to_inside"
            - "inside_to_outside"
            - "outside_to_inside"
            - "outside_to_outside"
        segments (list of str, optional): only if per_segment=True, 
            a list of the same four labels for each (points[i]→points[i+1]) transition.
    """
    x1, y1, x2, y2 = rect
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    def is_inside(pt):
        x, y = pt[0], pt[1]
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    if len(points) < 2:
        raise ValueError("Need at least two points to classify movement.")

    # overall
    start_in = is_inside(points[0])
    end_in   = is_inside(points[-1])
    if start_in and end_in:
        overall = "inside_to_inside"
    elif start_in and not end_in:
        overall = "inside_to_outside"
    elif not start_in and end_in:
        overall = "outside_to_inside"
    else:
        overall = "outside_to_outside"

    if not per_segment:
        return overall

    # per‐segment
    segs = []
    for prev, curr in zip(points, points[1:]):
        pi, ci = is_inside(prev), is_inside(curr)
        if pi and ci:
            segs.append("inside_to_inside")
        elif pi and not ci:
            segs.append("inside_to_outside")
        elif not pi and ci:
            segs.append("outside_to_inside")
        else:
            segs.append("outside_to_outside")
    return overall, segs



def get_rectangle_corners(x_min, y_min, x_max, y_max):
    # Deriving four corners from the (x_min, y_min, x_max, y_max)
    bottom_left = (x_min, y_min)
    bottom_right = (x_max, y_min)
    top_right = (x_max, y_max)
    top_left = (x_min, y_max)

    # Return the four corners
    return (bottom_left, bottom_right, top_right, top_left)

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def rectangle_area(points):
    if len(points) != 4:
        raise ValueError("Need exactly 4 points")

    # Assume points are ordered (clockwise or counterclockwise)
    side1 = distance(points[0], points[1])
    side2 = distance(points[1], points[2])

    return side1 * side2
def is_indide(roi, obj):
    status = False
    sx_min, sy_min, sx_max, sy_max = roi
    ox_min, oy_min, ox_max, oy_max = obj

    p_special_boundary = get_rectangle_corners(sx_min, sy_min, sx_max, sy_max)
    p_object_boundary = get_rectangle_corners(ox_min, oy_min, ox_max, oy_max)

    obj_boundary_list = list(p_object_boundary)
    obj_area = rectangle_area(obj_boundary_list)

    polygon_special_boundary = Polygon(p_special_boundary)
    polygon_object_boundary = Polygon(p_object_boundary)

    intersection_area = polygon_object_boundary.intersection(polygon_special_boundary).area

    ratio = intersection_area/obj_area
    if ratio >= 1/3:
        status =True
    return status



def process_video(input_path, output_path, model_path, target_class=0, conf=0.25, iou=0.5):
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Load YOLO model
    model = YOLO(model_path)
    item_names ={value: key for key, value in model.names.items()}
    print("Model class names:", model.names)


    video_writer_ = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    ROI = [(750, 1040), (750, 1740), (1490, 1740), (1490, 1040)]
    counter = solutions.ObjectCounter(
                    show=True,  # Display the image during processing
                    region=[(750, 1040), (750, 1740), (1490, 1740), (1490, 1040)],  # Region of interest points
                    model="/data/dataset/weights/base_weight/weights/best_wo_specialised_training.pt",  # Ultralytics YOLO11 model file
                    line_width=2,  # Thickness of the lines and bounding boxes
                    )
    max_items_count = {key: 0 for key in item_names.keys()}
    current_candidate_max = {key: 0 for key in item_names.keys()}  # Potential new maximum being verified
    consecutive_frames_count = {key: 0 for key in item_names.keys()}  # Counter for consecutive frames
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(frame, conf=conf, iou=iou, imgsz=2080, verbose=False)
        tracks = model.track(frame, persist=True, conf=0.55, iou=0.25, show=False, imgsz=2496, tracker="botsort.yaml")[0]
        if tracks.boxes is not None:
            boxes = tracks.boxes.xywh.cpu()
            classes = tracks.boxes.cls
            track_ids = tracks.boxes.id.int().cpu().tolist()

            for box,track_id, class_name in zip(boxes, track_ids,classes):
                x, y, w, h = box
                cx= (x+w)/2
                cy = (y+h)/2
                dx = max(750 - cx, cx - 1490, 0)
                dy = max(1040 - cy, cy - 1040, 0)
                i =0
                if (dx**2 + dy**2) <= 5000:
                    track = track_history[track_id]
                    track.append((float(x), float(y), class_name))  # x, y center point
                    if len(track) > 3:  # retain 30 tracks for 30 frames
                        track.pop(0)
                    x, y, obj_id = track[0]
                    if int(obj_id) == 13:
                        # 1) Round to integer pixel coords
                        cx, cy = int(x), int(y)

                        # 2) Draw a little filled circle (radius=5) in green
                        cv2.circle(
                            frame,            # your image
                            (cx, cy),         # center location
                            5,                # radius in pixels
                            (0, 255, 0),      # BGR color (green)
                            thickness=-1      # -1 for filled
                        )

                        # 3) Save the annotated frame (make sure 'i' is defined)
                        out_filename = f"frame_{i}.png"
                        cv2.imwrite(out_filename, frame)
                        try:
                            print(classify_track((750, 1040, 1490, 1040), track))
                        except:
                            pass
                    

        # print(track.boxes.cls, track.boxes.data, track.boxes.id, track.boxes.xywh)

        # results_ = counter(frame)
        # print(results_)
        # video_writer_.write(results_.plot_im)
        # Process detections
        items_count = {key:0 for key,_ in item_names.items()}
        for result in results:
            for box in result.boxes:
                if int(box.cls.item()) == target_class:
                    # Get and expand bounding box
                    bbox_xyxy = box.xyxy.cpu().numpy().squeeze().tolist()
                    adjusted_bbox = expand_bbox(bbox_xyxy, padding=100, 
                                              image_width=frame_width, 
                                              image_height=frame_height)
                    
                    # Draw rectangle on frame
                    x_min, y_min, x_max, y_max = map(int, adjusted_bbox)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                (0, 255, 0), 2)
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze())
                
                if is_indide(map(int, adjusted_bbox), map(int, box.xyxy.cpu().numpy().squeeze())):

                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                (0,0,255))
                    items_count[model.names[box.cls.item()]] +=1

        # Update maximum tracking logic
        for key in item_names.keys():
            current_count = items_count.get(key, 0)
            
            if current_candidate_max[key] > max_items_count[key]:
                # We have an active candidate being verified
                if current_count >= current_candidate_max[key]:
                    consecutive_frames_count[key] += 1
                    if consecutive_frames_count[key] >= 4:
                        # Update confirmed maximum
                        max_items_count[key] = current_candidate_max[key]
                        # Reset tracking variables
                        current_candidate_max[key] = 0
                        consecutive_frames_count[key] = 0
                else:
                    # Candidate not maintained, reset
                    current_candidate_max[key] = 0
                    consecutive_frames_count[key] = 0
            else:
                # Check for new potential maximum
                if current_count > max_items_count[key]:
                    current_candidate_max[key] = current_count
                    consecutive_frames_count[key] = 1


        frame = display_items_count(frame, max_items_count)

        # Write processed frame
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    model_path = "/data/dataset/weights/base_weight/weights/best_wo_specialised_training.pt"
    input_video = "/data/dataset/demo_video/final_210225/surg_sponge.avi"
    output_video = "./video_processed.mp4"
    
    process_video(
        input_path=input_video,
        output_path=output_video,
        model_path=model_path,
        target_class=0,
        conf=0.25,
        iou=0.5
    )