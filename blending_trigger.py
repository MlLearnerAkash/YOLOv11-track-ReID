from ultralytics import YOLO
import cv2
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import numpy as np


# Initialize model
model = YOLO('weights/best_wo_specialised_training.pt')
root_dir = "/Users/akashmanna/ws/opervu/demo_video/raw"

print(model.names)

video_paths = [join(root_dir,f) for f in listdir(root_dir) if isfile(join(root_dir, f)) and f.endswith((".mp4", ".avi"))]
print(video_paths)

# Initialize video captures
captures = [cv2.VideoCapture(path) for path in video_paths]

# Initialize trackers
trackers = [defaultdict(lambda: None) for _ in captures]

target_id = 10

primary_cam = 0

min_index = None
min_area = float("inf")
min_frame_obstructions = None  # Optionally store the obstructions for the minimum frame.


def calculate_total_area(detections):
    """
    Calculate the total area of detections.
    
    Each detection should be given as a list or array in the format:
    [x1, y1, x2, y2]
    
    The area is computed as:
        area = abs(x2 - x1) * abs(y2 - y1)
    
    Args:
        detections (array-like): A 2D array-like object where each row is a bounding box.
        
    Returns:
        float: The total area covered by all detections.
    """
    detections = np.array(detections)
    
    # Ensure each detection has 4 coordinates
    if detections.ndim != 2 or detections.shape[1] != 4:
        raise ValueError("Input detections should be a 2D array with shape (N, 4)")
    
    widths = np.abs(detections[:, 2] - detections[:, 0])
    heights = np.abs(detections[:, 3] - detections[:, 1])
    areas = widths * heights
    total_area = np.sum(areas)
    
    return total_area


def get_min_obstruction_frame(frames, model, target_id):
    """
    Processes the provided frames using the given model, computes the total obstruction area
    for each frame corresponding to the specified target_id, and returns the index of the frame
    with the least total obstruction area.
    
    Args:
        frames (list): List of frames (e.g., images read from cameras).
        model: A model that accepts a list of frames and returns results. 
               Each result must have:
                 - result.boxes.xywh (bounding boxes in [x, y, w, h] format)
                 - result.boxes.conf (detection confidences)
                 - result.boxes.cls (class IDs)
        target_id: The class ID corresponding to the obstruction.
    
    Returns:
        tuple: (min_index, min_frame_obstructions, min_area)
            - min_index (int): Index of the frame with the least total obstructions.
            - min_frame_obstructions (numpy.ndarray): Detections of obstructions in that frame.
            - min_area (float): The total obstruction area of that frame.
    """
    results = model(frames)  # Perform model inference on the list of frames.
    
    min_area = float("inf")
    min_index = None
    min_frame_obstructions = None

    for i, result in enumerate(results):
        # Extract detections, confidences, and class IDs from the result.
        detections = result.boxes.xywh.cpu().numpy()  # Expecting [x, y, w, h] format.
        confidences = result.boxes.conf.cpu().numpy()   # Not used here, but available.
        class_ids = result.boxes.cls.cpu().numpy()
        
        # Filter the detections that correspond to the target_id.
        obstructions = detections[np.where(class_ids == target_id)]
        total_obstructions = calculate_total_area(obstructions)
        print(f"Frame {i} total obstruction area: {total_obstructions}")

        # Check if this frame has less obstruction than the current minimum.
        if total_obstructions < min_area:
            min_area = total_obstructions
            min_index = i
            min_frame_obstructions = obstructions

    print(f"\nFrame with least total obstructions: {min_index} (Area: {min_area})")
    return min_index, min_frame_obstructions, min_area



while True:
    frames = [capture.read()[1] for capture in captures]
    primary_frame = frames[primary_cam]

    primary_result = model(primary_frame)

    if target_id in primary_result[0].boxes.cls.cpu().numpy():

        # results = model(frames)  # Single model inference

        # for i, result in enumerate(results):
        #     # Extract detections for the current frame
        #     detections = result.boxes.xywh.cpu().numpy()
        #     confidences = result.boxes.conf.cpu().numpy()
        #     class_ids = result.boxes.cls.cpu().numpy()

        #     # if target_id(obstruction class) in class_ids:
        #     obstructions = detections[np.where(class_ids==target_id)]
        #     total_obstructions = calculate_total_area(obstructions)
        #     print(total_obstructions)

        #     if total_obstructions < min_area:
        #         min_area = total_obstructions
        #         min_index = i
        #         min_frame_obstructions = obstructions

        # print(f"\nFrame with least total obstructions: {min_index} (Area: {min_area})")

        blend_primary_index, _, _ = get_min_obstruction_frame(frames=frames, model=model, 
                                  target_id=target_id)
        print(blend_primary_index)
