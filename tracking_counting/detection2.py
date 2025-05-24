import torch
import cv2
from ultralytics import YOLO
from draw_utils import display_items_count
from process_frame import process_frame

from ultralytics import solutions

from collections import defaultdict

import math
from shapely.geometry import Polygon
import numpy as np
from shapely.geometry import box


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

# def track_position_change(roi, first_pt, last_pt):
    
#     xmin, ymin, xmax, ymax = roi

#     @staticmethod
#     def is_inside(pt, xmin, xmax, ymin, ymax):
#         x, y,_,_ = pt
#         return xmin <= x <= xmax and ymin <= y <= ymax
#     # @staticmethod
#     # def is_inside(box_coords, xmin, xmax, ymin, ymax):
#     #     """
#     #     Check if a bounding box intersects with a rectangular region using Shapely.
        
#     #     Args:
#     #         box_coords: Tuple (x_center, y_center, width, height) of the detected object.
#     #         xmin, xmax, ymin, ymax: Coordinates of the predefined rectangular region.
            
#     #     Returns:
#     #         bool: True if the bounding box intersects the region.
#     #     """
#     #     # Convert detection box (center-based) to Shapely box (corner-based)
#     #     x_center, y_center, w, h = box_coords
#     #     detection_box = box(
#     #         x_center - w/2,  # Left
#     #         y_center - h/2,  # Top
#     #         x_center + w/2,  # Right
#     #         y_center + h/2   # Bottom
#     #     )
        
#     #     #Create predefined region as a Shapely box
#     #     region_box = box(xmin, ymin, xmax, ymax)
        
#     #     return detection_box.intersects(region_box)

#     first_inside = is_inside(first_pt,xmin, xmax, ymin, ymax)
#     last_inside  = is_inside(last_pt,xmin, xmax, ymin, ymax)

#     # if first_inside and last_inside:
#     #     return 'inside'
#     # elif not first_inside and not last_inside:
#         # return 'outside'
#     # if not first_inside and last_inside:
#     #     return 'entering'
#     if first_inside and not last_inside:  # first_inside and not last_inside
#         return 'exiting'
#     # if last_inside:
#     #     return "inside"
#     # if not last_inside:
#     #     return "outside"

#NOTE:v-2
def track_position_change(roi, track):
    xmin, ymin, xmax, ymax = roi
    
    if len(track) < 3:
        return None  # Not enough points to determine
    
    # Check last 4 points
    last_four = track[-3:]
    
    # Check if all last 4 points are exiting
    all_exiting = True
    for point in last_four:
        x, y, w, h = point[0]
        # Check if current point is outside ROI
        if not (x < xmin or x > xmax or y < ymin or y > ymax):
            all_exiting = False
            break
            
    # Check if any point before last 4 was inside
    was_inside = any(
        (xmin <= x <= xmax and ymin <= y <= ymax)
        for x, y, w, h in track[:-3].squeeze(1)
    )
    
    if all_exiting and was_inside:
        return 'exited'
    return None



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
    fps = 2
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Load YOLO model
    model = YOLO(model_path)
    item_names ={value: key for key, value in model.names.items()}
    print("Model class names:", model.names)


    
    max_items_count = {key: 0 for key in item_names.keys()}
    current_candidate_max = {key: 0 for key in item_names.keys()}  # Potential new maximum being verified
    consecutive_frames_count = {key: 0 for key in item_names.keys()}  # Counter for consecutive frames
    track_history = defaultdict(lambda: [])
    obj_history = defaultdict(list)

    # New trackers for track length
    max_track_length = defaultdict(int)            # confirmed maximum track length per key
    current_candidate_max_track = defaultdict(int) # candidate maximum track length per key
    consecutive_frames_track = defaultdict(int)    # consecutive frames count for track length candidates

    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = 2                               # frames-per-second of your output video
    height, width = 2048, 2048#frame.shape[:2]           # e.g. from your first frame
    out_tracking    = cv2.VideoWriter('output_tracking.mp4', fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        track_status = {}
        # Perform inference
        results = model.predict(frame, conf=conf, iou=iou, imgsz=2080, verbose=False)
        tracks = model.track(frame, persist=True, conf=0.5, iou=0.25, show=False, imgsz=2496, tracker="/home/opervu-user/opervu/ws/ultralytics/ultralytics/cfg/trackers/botsort.yaml")[0]
        if tracks.boxes is not None:
            boxes = tracks.boxes.xywh.cpu()
            classes = tracks.boxes.cls
            track_ids = tracks.boxes.id.int().cpu().tolist()
            
            track =[]
            
            x_min, x_max = 750, 1490-250
            y_min, y_max = 1040+250, 1740
            margin = 400
            max_track_len = 10

            # After updating track_history via your existing loop...
            for box, track_id, class_name in zip(boxes, track_ids, classes):
                x, y, w, h = box  # centers
                
                if class_name ==13:
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
                near_x = (x_min - margin/2 <= x < x_min + margin/2) or (x_max- margin/2 < x <= x_max + margin/2)
                near_y = (y_min - margin/2 <= y < y_min + margin/2) or (y_max- margin/2 < y <= y_max + margin/2)

                if near_x or near_y:
                    # ensure the list exists
                    track = track_history.setdefault(track_id, [])

                    track.append((float(x), float(y),float(w),float(h), class_name))
                    if len(track) > max_track_len:
                        track.pop(0)

            # Draw all trajectories from track_history
            item_status = []
            for track_id, track in track_history.items():
                
                
                # extract just the xy for polyline
                pts = np.array([(int(x), int(y), int(w), int(h)) for x, y, w, h, _ in track], np.int32).reshape(-1, 1, 4)
                if track[-1][-1] ==13:
                    draw_pts = np.array([(int(x), int(y)) for x, y, w, h, _ in track], np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [draw_pts], isClosed=False, color=(255,255,255), thickness=5)
                    # classification on the last point if desired
                    last_cls = track[-1][2]
                    label = f"ID {track_id}"
                    x_lbl, y_lbl = pts[-1][0][0] + 5, pts[-1][0][1] - 5
                    first_pts, last_pts = pts[0], pts[-1]
                    status = track_position_change((x_min, y_min, x_max, y_max), pts) #750, 1040, 1490, 1740,first_pts[0], last_pts[0]
                    if status:
                        item_status.append(status)
                        # Update track history to keep only last position
                        track_history[track_id] = [track[-1]]
                    if status:
                        cv2.putText(frame, status+label, (x_lbl, y_lbl),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                    # else:
                    #     cv2.putText(frame, label, (x_lbl, y_lbl),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            # Draw ROI
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Save or display
            out_tracking.write(frame)
        track_status[model.names[13]] = item_status 
        
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
            current_track_length = len(track_status.get(key, []))
            if current_candidate_max[key] > max_items_count[key]:
                # We have an active candidate being verified
                if current_count >= current_candidate_max[key]:
                    consecutive_frames_count[key] += 1
                    if consecutive_frames_count[key] >= 3:
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

            if current_track_length :
                max_items_count[key] = max_items_count[key] - current_track_length
            # --- Update logic for track length ---
        #     if current_candidate_max_track[key] > max_track_length[key]:
        #         # Verifying an active track-length candidate
        #         if current_track_length >= current_candidate_max_track[key]:
        #             consecutive_frames_track[key] += 1
        #             if consecutive_frames_track[key] >= 4:
        #                 # Confirm new max track length
        #                 max_track_length[key] = current_candidate_max_track[key]
        #                 # Reset track-length tracking
        #                 current_candidate_max_track[key] = 0
        #                 consecutive_frames_track[key] = 0
        #         else:
        #             # Candidate failed verification
        #             current_candidate_max_track[key] = 0
        #             consecutive_frames_track[key] = 0
        #     else:
        #         # Detect new potential track-length max
        #         if current_track_length > max_track_length[key]:
        #             current_candidate_max_track[key] = current_track_length
        #             consecutive_frames_track[key] = 1
        


        # #Update max_item_count
        # for key in item_names.keys():
        #     if max_items_count[key] or max_track_length[key]:
        #         max_items_count[key] = max_items_count[key] - max_track_length[key]
        #         #NOTE: Not working due to continuous update
        #         max_track_length[key] = 0



        # print(track_status)
        frame = display_items_count(frame, max_items_count, max_track_length)

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