import argparse
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
import json
from ultralytics import YOLO, solutions
from ultralytics.utils.files import increment_path

# Global mapping to maintain consistent names
object_id_to_name = {}
next_object_number = {}

def get_human_readable_name(obj_id, class_name):
    """
    Get or assign a human-readable name for the object.
    
    Args:
        obj_id: Unique ID of the object assigned by the tracker.
        class_name: The class name of the object.
        
    Returns:
        A human-readable name for the object.
    """
    if obj_id not in object_id_to_name:
        if class_name not in next_object_number:
            next_object_number[class_name] = 1
        object_id_to_name[obj_id] = f"{class_name}#{next_object_number[class_name]}"
        next_object_number[class_name] += 1
    return object_id_to_name[obj_id]

def is_inside_region(region_points, bbox):
    """
    Check if a bounding box is inside the specified region.
    
    Args:
        region_points: List of points defining the region [(x1, y1), (x2, y2), ...].
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        
    Returns:
        True if the bounding box is inside the region, False otherwise.
    """
    region_polygon = Polygon(region_points)
    bbox_polygon = Polygon([
        (bbox[0], bbox[1]),  # Top-left
        (bbox[2], bbox[1]),  # Top-right
        (bbox[2], bbox[3]),  # Bottom-right
        (bbox[0], bbox[3])   # Bottom-left
    ])
    return region_polygon.intersects(bbox_polygon)

# def run(
#     weights="/root/ws/med_si_track/custom_needle_large/large/weights/best.pt",
#     source="/root/ws/med_si_track/test_data/training_6.avi",
#     device="cpu",
#     view_img=False,
#     save_img=False,
#     exist_ok=False,
#     classes=None,
#     line_thickness=2,
#     track_thickness=2,
#     region_thickness=2,
# ):
#     model = YOLO(weights)
#     print("Model classes:", model.names)

#     cap = cv2.VideoCapture(source)
#     assert cap.isOpened(), "Error reading video file"
    
#     w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
#     # Define ROI points
#     line_points = [(750, 1040), (750, 1740), (1490, 1740), (1490, 1040)]
    
#     # Initialize ObjectCounter from ultralytics.solutions
#     counter = solutions.ObjectCounter(
#         view_img=view_img,
#         reg_pts=line_points,
#         classes_names=model.names,
#         draw_tracks=True,
#         line_thickness=line_thickness
#     )

#     # Initialize video writer if saving
#     if save_img:
#         save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
#         save_dir.mkdir(parents=True, exist_ok=True)
#         video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), 
#                                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#     i = 0
#     prev_tracks = {}  # To store the previous track info

#     while cap.isOpened():
#         success, im0 = cap.read()
#         if not success:
#             print("Video frame is empty or video processing has completed.")
#             break

#         # Perform object tracking
#         tracks = model.track(im0, persist=True, conf=0.55, iou=0.25, show=False, imgsz=2496, tracker="botsort.yaml", classes=[1])
#         im, im_status = counter.start_counting(im0, tracks)
#         print("<<<<<<<",tracks[0].boxes.id)
#         current_tracks = {}

#         if not im_status or len(im_status[0]) == 0:
#             print(f"No objects detected or tracked in frame {i}.")
#             if save_img:
#                 video_writer.write(im0)
#             continue

#         for track in tracks[0].boxes:
#             obj_id = int(track.id)  # Object ID
#             bbox = track.xyxy[0].tolist()  # Bounding box coordinates
#             class_id = int(track.cls[0])  # Class ID
#             class_name = model.names[class_id]

#             if is_inside_region(line_points, bbox) or obj_id in prev_tracks:
#                 # Maintain tracking within ROI or from previous frame
#                 human_readable_name = get_human_readable_name(obj_id, class_name)
#                 current_tracks[obj_id] = {"name": human_readable_name, "bbox": bbox}
        
#         prev_tracks = current_tracks

#         # Annotate frame
#         for obj_id, data in current_tracks.items():
#             bbox = data["bbox"]
#             cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#             cv2.putText(im0, data["name"], (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         # Save object data to JSON
#         with open(f"/root/ws/ultralytics/gui_data/item_{i}_status.json", "w") as f:
#             json.dump(current_tracks, f, indent=4)

#         cv2.imwrite(f"/root/ws/ultralytics/gui_data/item_{i}_status.jpg", im0)
#         i += 1

#         if save_img:
#             video_writer.write(im0)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     if save_img:
#         video_writer.release()
#     cv2.destroyAllWindows()

#-------------------------------------------------------------

# Import necessary libraries
import argparse
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
import json
from ultralytics import YOLO, solutions
from ultralytics.utils.files import increment_path

# Global mapping to maintain consistent names
object_id_to_name = {}
next_object_number = {}
original_id_map = {}
primitive_id_map = {}

def get_primitive_id(current_id):
    """
    Map the current ID to its original ID based on the ID switch map.
    Handles cycles gracefully to prevent infinite loops.
    """
    visited = set()  # To detect cycles
    while current_id in primitive_id_map:
        if current_id in visited:
            # Cycle detected; break by returning the current ID
            print(f"Cycle detected for ID {current_id}. Breaking cycle.")
            break
        visited.add(current_id)
        current_id = primitive_id_map[current_id]
    return current_id

def get_human_readable_name(obj_id, class_name):
    """
    Get or assign a human-readable name for the object.
    
    Args:
        obj_id: Unique ID of the object assigned by the tracker.
        class_name: The class name of the object.
        
    Returns:
        A human-readable name for the object.
    """
    if obj_id not in object_id_to_name:
        if class_name not in next_object_number:
            next_object_number[class_name] = 1
        object_id_to_name[obj_id] = f"{class_name}#{next_object_number[class_name]}"
        next_object_number[class_name] += 1
    return object_id_to_name[obj_id]

def is_inside_region(region_points, bbox):
    """
    Check if a bounding box is inside the specified region.
    
    Args:
        region_points: List of points defining the region [(x1, y1), (x2, y2), ...].
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        
    Returns:
        True if the bounding box is inside the region, False otherwise.
    """
    region_polygon = Polygon(region_points)
    bbox_polygon = Polygon([
        (bbox[0], bbox[1]),  # Top-left
        (bbox[2], bbox[1]),  # Top-right
        (bbox[2], bbox[3]),  # Bottom-right
        (bbox[0], bbox[3])   # Bottom-left
    ])
    return region_polygon.intersects(bbox_polygon)

def run(
    weights="/root/ws/med_si_track/custom_needle_large/large/weights/best.pt",
    source="/root/ws/med_si_track/test_data/training_6.avi",
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    global primitive_id_map  # Ensure primitive_id_map is accessible
    model = YOLO(weights)
    print("Model classes:", model.names)

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), "Error reading video file"
    
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    # Define ROI points
    line_points = [(750, 1040), (750, 1740), (1490, 1740), (1490, 1040)]
    
    # Initialize ObjectCounter from ultralytics.solutions
    counter = solutions.ObjectCounter(
        view_img=view_img,
        reg_pts=line_points,
        classes_names=model.names,
        draw_tracks=True,
        line_thickness=line_thickness
    )

    # Initialize video writer if saving
    if save_img:
        save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), 
                                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    i = 0
    prev_tracks = {}  # To store persistent tracking info
    current_object_numbers = {}  # To track contiguous numbers for each class
    id_switch_map = {}  # To track ID switches

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has completed.")
            break

        # Perform object tracking
        tracks = model.track(im0, persist=True, conf=0.55, iou=0.25, show=False, imgsz=2496, tracker="botsort.yaml", classes=[1])
        im, im_status = counter.start_counting(im0, tracks)
        if not im_status or len(im_status[0]) == 0:
            print(f"No objects detected or tracked in frame {i}.")
            if save_img:
                video_writer.write(im0)
            continue

        # Update tracking information
        current_tracks = {}

        for track in tracks[0].boxes:
            obj_id = int(track.id)  # Object ID
            bbox = track.xyxy[0].tolist()  # Bounding box coordinates
            class_id = int(track.cls[0])  # Class ID
            class_name = model.names[class_id]

            # Detect ID switch
            if obj_id not in prev_tracks:
                for prev_id in prev_tracks:
                    if prev_tracks[prev_id]["class_name"] == class_name and is_inside_region(line_points, bbox):
                        id_switch_map[prev_id] = obj_id
                        primitive_id_map[obj_id] = get_primitive_id(prev_id)
                        break

            # Get primitive ID for the current object
            primitive_id = get_primitive_id(obj_id)
            if is_inside_region(line_points, bbox) or primitive_id in prev_tracks:
                if primitive_id not in prev_tracks:
                    # Assign a contiguous number for the class
                    if class_name not in current_object_numbers:
                        current_object_numbers[class_name] = 1
                    object_id_to_name[primitive_id] = f"{class_name}#{current_object_numbers[class_name]}"
                    current_object_numbers[class_name] += 1

                human_readable_name = object_id_to_name[primitive_id]
                current_tracks[obj_id] = {"name": human_readable_name, "bbox": bbox, "class_name": class_name}
        
        prev_tracks = current_tracks

        # Annotate frame
        for obj_id, data in current_tracks.items():
            bbox = data["bbox"]
            cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(im0, data["name"], (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save object data to JSON
        json_data = {
            "frame": i,
            "objects": [
                {
                    "id": obj_id,
                    "name": data["name"],
                    "primitive_id": get_primitive_id(obj_id),
                    "bbox": data["bbox"],
                    "class_name": data["class_name"],
                    "is_inside": is_inside_region(line_points, data["bbox"]),
                }
                for obj_id, data in current_tracks.items()
            ],
            "id_switches": id_switch_map
        }
        with open(f"/root/ws/ultralytics/gui_data/item_{i}_status.json", "w") as f:
            json.dump(json_data, f, indent=4)

        cv2.imwrite(f"/root/ws/ultralytics/gui_data/item_{i}_status.jpg", im0)
        i += 1

        if save_img:
            video_writer.write(im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_img:
        video_writer.release()
    cv2.destroyAllWindows()

#-------------------------------------------------------------

# import argparse
# from pathlib import Path
# import cv2
# import numpy as np
# from shapely.geometry import Polygon
# import json
# from ultralytics import YOLO, solutions
# from ultralytics.utils.files import increment_path

# # Global mapping to maintain consistent names and ID history
# object_id_to_name = {}
# original_id_map = {}
# next_object_number = {}

# def get_human_readable_name(obj_id, class_name):
#     """
#     Get or assign a human-readable name for the object.
#     """
#     if obj_id not in object_id_to_name:
#         if class_name not in next_object_number:
#             next_object_number[class_name] = 1
#         object_id_to_name[obj_id] = f"{class_name}#{next_object_number[class_name]}"
#         next_object_number[class_name] += 1
#     return object_id_to_name[obj_id]

# def is_inside_region(region_points, bbox):
#     """
#     Check if a bounding box is inside the specified region.
#     """
#     region_polygon = Polygon(region_points)
#     bbox_polygon = Polygon([
#         (bbox[0], bbox[1]),  # Top-left
#         (bbox[2], bbox[1]),  # Top-right
#         (bbox[2], bbox[3]),  # Bottom-right
#         (bbox[0], bbox[3])   # Bottom-left
#     ])
#     return region_polygon.intersects(bbox_polygon)

# def map_original_id(current_id):
#     """
#     Map the current ID to its original ID based on the ID switch map.
#     Handles cycles gracefully to prevent infinite loops.
#     """
#     visited = set()  # To detect cycles
#     while current_id in original_id_map:
#         if current_id in visited:
#             # Cycle detected; break by returning the current ID
#             print(f"Cycle detected for ID {current_id}. Breaking cycle.")
#             break
#         visited.add(current_id)
#         current_id = original_id_map[current_id]
#     return current_id


# def run(
#     weights="/root/ws/med_si_track/custom_needle_large/large/weights/best.pt",
#     source="/root/ws/med_si_track/test_data/training_6.avi",
#     device="cpu",
#     view_img=False,
#     save_img=False,
#     exist_ok=False,
#     classes=None,
#     line_thickness=2,
#     track_thickness=2,
#     region_thickness=2,
#     max_undetected_frames=6,  # Drop objects after 60 frames
# ):
#     model = YOLO(weights)
#     print("Model classes:", model.names)

#     cap = cv2.VideoCapture(source)
#     assert cap.isOpened(), "Error reading video file"
    
#     w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
#     # Define ROI points
#     line_points = [(750, 1040), (750, 1740), (1490, 1740), (1490, 1040)]
    
#     # Initialize ObjectCounter from ultralytics.solutions
#     counter = solutions.ObjectCounter(
#         view_img=view_img,
#         reg_pts=line_points,
#         classes_names=model.names,
#         draw_tracks=True,
#         line_thickness=line_thickness
#     )

#     # Initialize video writer if saving
#     if save_img:
#         save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
#         save_dir.mkdir(parents=True, exist_ok=True)
#         video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), 
#                                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#     frame_count = 0
#     prev_tracks = {}
#     undetected_objects = {}
#     inside_count = 0
#     outside_count = 0

#     while cap.isOpened():
#         success, im0 = cap.read()
#         if not success:
#             print("Video frame is empty or video processing has completed.")
#             break

#         # Perform object tracking
#         tracks = model.track(im0, persist=True, conf=0.55, iou=0.25, show=False, imgsz=2496, tracker="botsort.yaml", classes=[1])
#         im, im_status = counter.start_counting(im0, tracks)
#         current_tracks = {}
        
#         if im_status and len(im_status[0]) > 0:
#             for track in tracks[0].boxes:
#                 obj_id = int(track.id)  # Object ID
#                 bbox = track.xyxy[0].tolist()  # Bounding box coordinates
#                 class_id = int(track.cls[0])  # Class ID
#                 class_name = model.names[class_id]
                
#                 # Map to original ID if an ID switch is detected
#                 if obj_id not in original_id_map:
#                     for prev_id, prev_data in prev_tracks.items():
#                         if prev_data["class_name"] == class_name and is_inside_region(line_points, bbox):
#                             original_id_map[obj_id] = map_original_id(prev_id)
#                             break
#                 original_id = map_original_id(obj_id)
#                 human_readable_name = get_human_readable_name(original_id, class_name)
#                 current_tracks[obj_id] = {
#                     "name": human_readable_name,
#                     "bbox": bbox,
#                     "class_name": class_name,
#                     "original_id": original_id,
#                 }

#                 # Remove from undetected objects if found
#                 if original_id in undetected_objects:
#                     del undetected_objects[original_id]
            
#                 # Update inside/outside count
#                 if is_inside_region(line_points, bbox):
#                     inside_count += 1
#                 else:
#                     outside_count += 1

#         # Handle undetected objects
#         for obj_id, data in prev_tracks.items():
#             original_id = map_original_id(obj_id)
#             if original_id not in current_tracks:
#                 if original_id not in undetected_objects:
#                     undetected_objects[original_id] = {
#                         "name": data["name"],
#                         "bbox": data["bbox"],
#                         "class_name": data["class_name"],
#                         "undetected_frames": 0,
#                     }
#                 undetected_objects[original_id]["undetected_frames"] += 1

#         # Drop objects undetected for more than max_undetected_frames
#         undetected_objects = {
#             obj_id: data
#             for obj_id, data in undetected_objects.items()
#             if data["undetected_frames"] <= max_undetected_frames
#         }
        
#         # Check for differences between inside and outside counts
#         if inside_count > outside_count:
#             print("Difference detected: More objects inside than outside.")

#         # Merge detected and undetected objects
#         all_objects = {
#             **{
#                 obj_id: {
#                     "name": data["name"],
#                     "bbox": data["bbox"],
#                     "class_name": data["class_name"],
#                     "detected": True,
#                 }
#                 for obj_id, data in current_tracks.items()
#             },
#             **{
#                 obj_id: {
#                     "name": data["name"],
#                     "bbox": data["bbox"],
#                     "class_name": data["class_name"],
#                     "detected": False,
#                 }
#                 for obj_id, data in undetected_objects.items()
#             },
#         }

#         # Annotate frame
#         # for obj_id, data in all_objects.items():
#         #     bbox = data["bbox"]
#         #     color = (0, 255, 0) if data["detected"] else (0, 0, 255)
#         #     cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
#         #     cv2.putText(im0, data["name"], (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         # Save frame data to JSON
#         json_data = {
#             "frame": frame_count,
#             "objects": [
#                 {
#                     "id": obj_id,
#                     "name": data["name"],
#                     "bbox": data["bbox"],
#                     "class_name": data["class_name"],
#                     "detected": data["detected"],
#                 }
#                 for obj_id, data in all_objects.items()
#             ],
#         }
#         with open(f"/root/ws/ultralytics/gui_data/frame_{frame_count}_data.json", "w") as f:
#             json.dump(json_data, f, indent=4)

#         if save_img:
#             video_writer.write(im0)
#         cv2.imwrite(f"/root/ws/ultralytics/gui_data/frame_{frame_count}.jpg", im0)
#         frame_count += 1

#         prev_tracks = current_tracks

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     if save_img:
#         video_writer.release()
#     cv2.destroyAllWindows()



def parse_opt():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/root/ws/med_si_track/custom_needle/train7/weights/best.pt", help="model weights path")
    parser.add_argument("--source", type=str, default="/root/ws/med_si_track/test_data/training_6.avi", help="path to video file")
    parser.add_argument("--device", type=str, default="cpu", help="device to use (cpu or cuda)")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="overwrite existing results")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=2, help="region boundary thickness")
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
