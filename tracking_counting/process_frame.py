import cv2
import numpy as np

def process_frame(
    frame: np.ndarray,
    model,
    target_class: int,
    item_names: dict,
    conf: float = 0.25,
    iou: float = 0.45,
    padding: int = 100,
    imgsz: int = 2080
) -> np.ndarray:
    
    # Frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Run inference
    results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)

    # Initialize counts
    items_count = {name: 0 for _, name in item_names.items()}

    # We'll store the expanded bbox from the first target_class detection
    expanded = None

    # 1) Find & expand the target-class bbox
    for res in results:
        for box in res.boxes:
            if int(box.cls.item()) == target_class:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().squeeze().tolist()
                expanded = expand_bbox(
                    [x1, y1, x2, y2],
                    padding=padding,
                    image_width=frame_width,
                    image_height=frame_height
                )
                # draw expanded bbox
                ex1, ey1, ex2, ey2 = map(int, expanded)
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
                # assume only one such region; break out once found
                break
        if expanded is not None:
            break

    if expanded is None:
        # no target-class found, just return frame
        return frame

    ex1, ey1, ex2, ey2 = map(int, expanded)

    # 2) Draw all detections inside that expanded bbox
    for res in results:
        for box in res.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy.cpu().numpy().squeeze())
            if bx1 >= ex1 and by1 >= ey1 and bx2 <= ex2 and by2 <= ey2:
                cls_idx = int(box.cls.item())
                name = model.names[cls_idx]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                items_count[name] = items_count.get(name, 0) + 1

    # 3) Overlay the counts on the frame
    frame = display_items_count(frame, items_count)

    return frame
