from ultralytics import YOLO
import wandb
# from wandb.integration.ultralytics import add_wandb_callback
import torch



# Load a model
model = YOLO("/mnt/data/weights/opervu_28SIs_07032025/opervu_28SIs_07032025_/weights/best.pt")


print("Updatedmodelnames>>>>",model.names)
validation_results = model.val(data="/mnt/data/dataset/YOLODataset/dataset.yaml", imgsz=2048, 
                               batch=1, conf=0.15, iou=0.25, 
                               device="0", plots = True,
                               save_json= True, #classes=[2]
                               )


# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os

# def calculate_iou(box1, box2):
#     """Calculate Intersection over Union (IoU) between two boxes"""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
    
#     inter_area = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
#     return inter_area / (box1_area + box2_area - inter_area + 1e-6)

# # Load trained model
# model = YOLO("/mnt/data/weights/opervu_needle_dark_removed_27012025/opervu_needle_dark_removed_27012025_/weights/best.pt")
# print("Updated model names>>>>", model.names)

# # Create output directory
# os.makedirs("annotated_results", exist_ok=True)

# # Run validation and get predictions
# results = model.val(
#     data="/mnt/data/dataset/YOLODataset/dataset.yaml",
#     imgsz=64,
#     batch=1,
#     conf=0.15,
#     iou=0.25,
#     device="0",
#     save_json=True,
# )
# print(results)
# # Process each image to annotate FP/FN
# for image_result in results:
#     # Load original image
#     img_path = image_result.path
#     image = cv2.imread(img_path)
    
#     # Get predictions and ground truth
#     pred_boxes = image_result.boxes.xyxy.cpu().numpy()
#     gt_boxes = image_result.boxes.xywhn  # Ground truth in normalized format
    
#     # Convert ground truth to absolute coordinates
#     height, width = image.shape[:2]
#     gt_boxes_abs = []
#     for box in gt_boxes:
#         x_center = box[0] * width
#         y_center = box[1] * height
#         w = box[2] * width
#         h = box[3] * height
#         gt_boxes_abs.append([
#             x_center - w/2,  # x1
#             y_center - h/2,  # y1
#             x_center + w/2,  # x2
#             y_center + h/2   # y2
#         ])
    
#     # Identify FP and FN
#     fp_boxes = []
#     fn_boxes = []
    
#     # Check for False Positives
#     for pred_box in pred_boxes:
#         is_fp = True
#         for gt_box in gt_boxes_abs:
#             if calculate_iou(pred_box[:4], gt_box) > 0.25:
#                 is_fp = False
#                 break
#         if is_fp:
#             fp_boxes.append(pred_box)
    
#     # Check for False Negatives
#     for gt_box in gt_boxes_abs:
#         is_fn = True
#         for pred_box in pred_boxes:
#             if calculate_iou(pred_box[:4], gt_box) > 0.25:
#                 is_fn = False
#                 break
#         if is_fn:
#             fn_boxes.append(gt_box)
    
#     # Draw annotations
#     # Draw FP in red
#     for box in fp_boxes:
#         x1, y1, x2, y2 = map(int, box[:4])
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
#         cv2.putText(image, 'FP', (x1, y1-10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
#     # Draw FN in blue
#     for box in fn_boxes:
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
#         cv2.putText(image, 'FN', (x1, y1-10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
#     # Save annotated image
#     output_path = os.path.join("annotated_results", os.path.basename(img_path))
#     cv2.imwrite(output_path, image)

# print("Annotation complete. Results saved in 'annotated_results' directory")