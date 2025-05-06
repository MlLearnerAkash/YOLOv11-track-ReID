import torch
import cv2
from ultralytics import YOLO
from draw_utils import display_items_count
from process_frame import process_frame
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

    max_items_count = {key: 0 for key in item_names.keys()}
    current_candidate_max = {key: 0 for key in item_names.keys()}  # Potential new maximum being verified
    consecutive_frames_count = {key: 0 for key in item_names.keys()}  # Counter for consecutive frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(frame, conf=conf, iou=iou, imgsz=2080, verbose=False)
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

                if (x1 >= x_min and y1 >= y_min and 
                    x2 <= x_max and y2 <= y_max):

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
    input_video = "/data/dataset/demo_video/gauze-needle.mp4"
    output_video = "./video_processed.mp4"
    
    process_video(
        input_path=input_video,
        output_path=output_video,
        model_path=model_path,
        target_class=0,
        conf=0.25,
        iou=0.5
    )