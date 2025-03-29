import cv2
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_yolo_model(model_path='yolov5s.pt'):
    model = torch.load(model_path)["model"]
    model.eval()
    return model


def preprocess_image(image_path, img_size=(2048, 2048)):
    # Load and resize image
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)

    # Convert BGR to RGB and to tensor format [C, H, W]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    # img /= 255.0  # Normalize to [0, 1]
    return torch.tensor(img[None])  # Add batch dimension


def load_yolo_annotations(anno_path, img_width, img_height):
    boxes = []
    with open(anno_path, 'r') as f:
        for line in f.readlines():
            
            # Split line and parse class ID and coordinates
            values = line.strip().split()
            class_id = float(values[0])
            coords = list(map(float, values[1:]))  # Remaining values are coordinates
            
            # Convert coordinates to absolute pixel values and create polygon
            polygon = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i + 1] * img_height)
                polygon.append((x, y))

            # Convert polygon to bounding box
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Store bounding box as (class_id, x_min, y_min, x_max, y_max)
            boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes

def draw_boxes(image, boxes, color, label_text):
    img = image.copy()
    for box in boxes:
        class_id, x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def compute_false_positives_negatives(gt_boxes, pred_boxes, iou_threshold=0.5):
    false_negatives, false_positives = [], []
    
    # Check if predictions overlap sufficiently with ground truth to classify correctly
    for gt in gt_boxes:
        matched = False
        for pred in pred_boxes:
            iou = calculate_iou(gt[1:], pred[1:])
            if iou >= iou_threshold:
                matched = True
                break
        if not matched:
            false_negatives.append(gt)

    for pred in pred_boxes:
        matched = False
        for gt in gt_boxes:
            iou = calculate_iou(gt[1:], pred[1:])
            if iou >= iou_threshold:
                matched = True
                break
        if not matched:
            false_positives.append(pred)

    return false_negatives, false_positives

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def main(model_path, image_path, gt_path):
    model = load_yolo_model(model_path).to(device)
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Load GT and Prediction Annotations
    gt_boxes = load_yolo_annotations(gt_path, img_width, img_height)
    results = model(preprocess_image(image_path).to(device).half())
    
    # Convert results to numpy and select only the detection with the highest score (4th position)
    preds = results[0].cpu().numpy()
    highest_score_pred = max(preds, key=lambda x: x[3])[0]  # Assuming x[3] is the score
    print(highest_score_pred)

    # Log the selected prediction
    pred_boxes = [(highest_score_pred[-2], int(highest_score_pred[0]), int(highest_score_pred[1]), 
                   int(highest_score_pred[2]), int(highest_score_pred[3]))]

    print("Highest score prediction:")
    print(f"Class ID: {pred_boxes[0][0]}, Box: ({pred_boxes[0][1]}, {pred_boxes[0][2]}, {pred_boxes[0][3]}, {pred_boxes[0][4]})")


    # Compute false positives and false negatives
    false_negatives, false_positives = compute_false_positives_negatives(gt_boxes, pred_boxes)
    
    # Draw boxes on images
    img_gt = draw_boxes(img, gt_boxes, (0, 255, 0), "GT")        # Green for ground truth
    img_fn = draw_boxes(img, false_negatives, (0, 0, 255), "FN") # Red for false negatives
    img_fp = draw_boxes(img, false_positives, (255, 0, 0), "FP") # Blue for false positives
    
    # Horizontally stack images (GT | FN | FP)
    output_image = np.hstack((img_gt, img_fn, img_fp))
    cv2.imwrite("GtVsFnVsFp.png", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    main('/root/ws/HIC-Yolov5/runs/train/only_needle_images/weights/best.pt', '/mnt/data/needle_images_only/needle_images/3446.jpg', '/mnt/data/needle_images_only/needle_images/3446.txt')
    # print(load_yolo_model("/root/ws/HIC-Yolov5/runs/train/only_needle_images/weights/best.pt")["model"].eval())