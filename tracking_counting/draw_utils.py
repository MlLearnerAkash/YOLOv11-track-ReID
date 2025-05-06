import os
import sys
import cv2


def draw_bbox_on_image(image, bbox, color=(0, 255, 0), thickness=2):

    # Create a copy of the image to avoid modifying the original
    image_copy = image.copy()
    
    # Convert coordinates to integers
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    # Draw the rectangle
    cv2.rectangle(
        img=image_copy,
        pt1=(x_min, y_min),
        pt2=(x_max, y_max),
        color=color,
        thickness=thickness
    )
    
    return image_copy

def display_items_count(frame, items_count, right_margin=10, y_start=30, line_spacing=30, 
                        font_scale=0.7, color=(0, 255, 0), thickness=2):
    """
    Display key-value counts on the top-right corner of a frame.
    
    """
    
    frame_height, frame_width = frame.shape[:2]
    
    for i, (key, value) in enumerate(items_count.items()):
        text = f"{key}: {value}"
        text_x = frame_width - right_margin - 200  # Adjust 200 for your text width
        text_y = y_start + i * line_spacing
        
        cv2.putText(
            img=frame,
            text=text,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=thickness
        )
    return frame