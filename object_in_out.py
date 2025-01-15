# import ultralytics
# ultralytics.checks()

# import cv2
# from ultralytics import YOLO, solutions


# # Load the pre-trained YOLOv8 model
# model = YOLO("/root/ws/med_si_track/custom_needle/train7/weights/best.pt")
# print(model.names)
# # Open the video file
# cap = cv2.VideoCapture("/root/ws/med_si_track/test_data/training_6.avi")
# assert cap.isOpened(), "Error reading video file"

# # Get video properties: width, height, and frames per second (fps)
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Define points for a line or region of interest in the video frame
# line_points = [(20, 20),(20, 2048), (2048, 2048), (2048, 20)]  # Line coordinates

# # Specify classes to count, for example: person (0) and car (2)
# classes_to_count = [0] # Class IDs for person and car

# # Initialize the video writer to save the output video
# video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# # Initialize the Object Counter with visualization options and other parameters
# counter = solutions.ObjectCounter(
#     view_img=True,  # Display the image during processing
#     reg_pts=line_points,  # Region of interest points
#     classes_names=model.names,  # Class names from the YOLO model
#     draw_tracks=True,  # Draw tracking lines for objects
#     line_thickness=4,  # Thickness of the lines drawn
# )


# # Process video frames in a loop
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     # Perform object tracking on the current frame, filtering by specified classes
#     tracks = model.track(im0, persist=True, conf=0.1, iou=0.1, show=False, classes=classes_to_count, imgsz = 2480)

#     # Use the Object Counter to count objects in the frame and get the annotated image
#     im0 = counter.start_counting(im0, tracks)

#     # Write the annotated frame to the output video
#     video_writer.write(im0)

# # Release the video capture and writer objects
# cap.release()
# video_writer.release()

# # Close all OpenCV windows
# cv2.destroyAllWindows()


'''
{0: 'clamp', 1: 'sponge', 2: 'gauze', 3: 'glove', 4: 'snare', 5: 'incision', 6: 'forceps', 
7: 'obstruction', 8: 'vesiloop', 9: 'sucker', 10: 'black_suture', 11: 'scissors', 12: 'bovie', 13: 
'woodspack', 14: 'needle', 15: 'needle_holder', 16: 'woods_pack', 17: 'scalpel'}
'''

import argparse
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO, solutions
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

import json

def mouse_callback(event, x, y, flags, param):
    """Mouse event callback for manipulating regions."""
    pass  # Keep the mouse callback if required later

def is_point_in_rectangle(rect_coords, point_coords):
    """
    Checks if a point is inside a rectangle defined by four corner coordinates.

    Args:
        rect_coords: A list of four coordinates representing the rectangle corners 
                    in any order: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        point_coords: A tuple representing the point coordinates (x, y).

    Returns:
        True if the point is inside the rectangle, False otherwise.
    """

    # Extract x and y coordinates from rectangle corners
    x_coords = [x for x, y in rect_coords]
    y_coords = [y for x, y in rect_coords]

    # Calculate min and max x and y
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    x, y = point_coords[0]

    # Check if the point is within the bounds
    return min_x <= x <= max_x and min_y <= y <= max_y

max_in_value = float('-inf')  # Initialize to negative infinity to ensure it works for any number
max_out_value = float('-inf')  # Initialize to negative infinity to ensure it works for any number'
# Function to update the maximum value
def update_in_max(value):
    global max_in_value
    max_in_value = max(max_in_value, value)  # Update max_value if value is greater
def update_out_max(value):
    global max_out_value
    max_out_value = max(max_out_value, value)  # Update max_value if value is greater

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
    """
    Run YOLOv8 with object counting and tracking using the ultralytics `solutions.ObjectCounter`.

    Args:
        weights (str): Path to model weights.
        source (str): Path to the video file.
        device (str): Device to use: 'cpu' or 'cuda'.
        view_img (bool): Whether to show the video while processing.
        save_img (bool): Whether to save the annotated output video.
        exist_ok (bool): Whether to overwrite existing output.
        classes (list): Classes to detect and track.
        line_thickness (int): Line thickness for annotations.
        track_thickness (int): Line thickness for tracking lines.
        region_thickness (int): Thickness of region boundary lines.
    """
    # Check environment and dependencies

    # Initialize the YOLO model
    model = YOLO(weights)
    print("Model classes:", model.names)

    # Load video
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), "Error reading video file"
    
    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define points for a region of interest (ROI) or line
    # line_points = [(350, 20), (350, 1040), (1490, 1040), (1490, 20)]  # ROI coordinates
    line_points = [(750, 1040), (750, 1740), (1490, 1740), (1490, 1040)]
    # Define object classes to count (use class IDs)
    classes_to_count = []  # For example, counting class 0 (persons, etc.)

    # Initialize video writer if saving
    if save_img:
        save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), 
                                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize ObjectCounter from ultralytics.solutions
    counter = solutions.ObjectCounter(
        view_img=view_img,  # Display the video frame while processing
        reg_pts=line_points,  # ROI for counting objects
        classes_names=model.names,  # YOLO class names
        draw_tracks=True,  # Draw tracking lines on objects
        line_thickness=line_thickness  # Line thickness for tracking
    )

    # Process video frame-by-frame
    i= 0
    
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has completed.")
            break
        global_in_count = 0
        global_out_count = 0
        # Perform object tracking with YOLO, filtering by specified classes
        tracks = model.track(im0, persist=True, conf=0.55,iou=0.25, show=False, imgsz=2496, tracker="botsort.yaml",classes=[1])#, 
        # Convert bounding boxes to center points
        # if tracks[0]:
        #     for k in range(len(tracks[0].boxes.xyxy)):
        #         center_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in tracks[0].boxes[k].xyxy]
        #         # Check if any center point is inside the rectangle
        #         if is_point_in_rectangle(line_points, center_points):
        #             # Increment the count for objects entering the rectangle
        #             global_in_count += 1
        #             update_in_max(global_in_count)

        #         else:
        #             # Increment the count for objects entering the rectangle
        #             global_out_count += 1
        #             # print(f"Object entered the rectangle. Total count: {global_out_count}")
        #             # out_current_status = max(out, global_out_count)
        #             update_out_max(global_out_count)
        # # print(f"Total objects entered the rectangle: {global_in_count}")
        # # 
        # try:
        #     print(f">>>>>>>>>>Numbe of Sponges inside incision {max_in_value}")
        #     print(f">>>>>>>>>>Numbe of Sponges outside incision: {max_out_value}")
        # except:
        #     pass
        # Count objects in the current frame using the ObjectCounter
        im0, item_status = counter.start_counting(im0, tracks)
        print(">>>>>", item_status)
        item_status_global= {}
        for item_num in range(len(item_status)):
            item_name= item_status[item_num]["class_name"]
            if item_name not in item_status_global.keys():
                item_status_global[item_name] = []
                
            item_status_global[item_name].append({"name": f"{item_name}#{item_num}", "status": f"{item_status[item_num]['is_inside']}"})

        print(">>>>>>",item_status_global)    
         
        # Write the updated all_item_status to a JSON file
        with open(f"/root/ws/ultralytics/gui_data/item_{i}_status.json", "w") as f:
            #TODO: Add the "item_status" to the JSON file
            json.dump(item_status_global, f, indent=4)

        #writting the image
        cv2.imwrite(f"/root/ws/ultralytics/gui_data/item_{i}_status.jpg", im0)
        
        i+=1

        # Save the annotated frame to output video
        if save_img:
            video_writer.write(im0)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    if save_img:
        video_writer.release()
    cv2.destroyAllWindows()
    


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
