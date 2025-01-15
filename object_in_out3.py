import argparse
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
import json
from ultralytics import YOLO, solutions
from ultralytics.utils.files import increment_path

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
    global_map = {}  # To store global IDs
    max_det_ = None
    while cap.isOpened():
        
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has completed.")
            break
        print(f">>>>>>>>>global_map: {global_map}")

        

        # Perform object tracking
        tracks = model.track(im0, persist=True, conf=0.55, iou=0.25, show=False, imgsz=2496, tracker="botsort.yaml", classes=[1])
        im, im_status = counter.start_counting(im0, tracks)
        i += 1
        if not im_status or len(im_status[0]) == 0:
            print(f"No objects detected or tracked in frame {i}.")
            global_map[i] = global_map[i - 1] if i > 1 else {}
            cv2.imwrite(f"/root/ws/ultralytics/gui_data/frame_{i}.jpg", im)
            with open(f"/root/ws/ultralytics/gui_data/frame_{i}.json", "w") as f:
                json.dump(global_map[i], f, indent=4)
            if save_img:
                video_writer.write(im0)
            
            continue

        local_map = {}
        for track in tracks[0].boxes:
            l =[]
            obj_id = int(track.id)  # Object ID
            bbox = track.xyxy[0].tolist()  # Bounding box coordinates
            class_id = int(track.cls[0])  # Class ID
            class_name = model.names[class_id]
            is_inside = is_inside_region(line_points, bbox)
            val = {"id": obj_id, "inside": is_inside}
            # Initialize the list for the class_name key if it doesn't exist
            if class_name not in local_map:
                local_map[class_name] = []
            
            # Append the value to the class_name list
            local_map[class_name].append(val)


        global_map[i] = local_map #{"id": obj_id,  "inside": is_inside}#"class": class_name, "bbox": bbox,

        for class_name, item_list in global_map[i].items():

            total_items = len(item_list)  # Total number of items
            inside_true_count = sum(1 for item in item_list )  # Count where 'inside' is True #```if item['inside']```

        if inside_true_count >1 and total_items == inside_true_count:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",i)
            max_det = global_map[i]
            max_det_ = max_det
        else:
            # Default behavior when condition is not satisfied
            max_det = {class_name: [] for class_name in global_map[i].keys()}  # Initialize empty for each class

        print(f"Frame {i}: {max_det_}")  

        #NOTE: Append the max_det to the global_map according to different classes
        # if len(global_map[i][class_name]) < len(max_det.values()):
        #     print("$$$$$$$$$$$$$$$$$$$$$$", i)
        #     global_map[i][class_name].append(max_det.values()[-1])

        if max_det_:
            # print(">>>>>>>>>",max_det_[class_name])
            global_map[i][class_name].append(max_det_[class_name][-1])
            # if max_det.get(class_name):  # Ensure `class_name` exists in `max_det`
            #     global_map[i][class_name].append(list(max_det[class_name])[-1] if max_det[class_name] else None)
        

        with open(f"/root/ws/ultralytics/gui_data/frame_{i}.json", "w") as f:
            json.dump(global_map[i], f, indent=4)

        cv2.imwrite(f"/root/ws/ultralytics/gui_data/frame_{i}.jpg", im)
            


        

        # cv2.imwrite(f"/root/ws/ultralytics/gui_data/item_{i}_status.jpg", im0)
        

        if save_img:
            video_writer.write(im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_img:
        video_writer.release()
    cv2.destroyAllWindows()

#-


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
