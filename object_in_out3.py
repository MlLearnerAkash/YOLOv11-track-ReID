import argparse
from pathlib import Path
import cv2
from collections import OrderedDict
import numpy as np
from shapely.geometry import Polygon
import json
from ultralytics import YOLO, solutions
from ultralytics.utils.files import increment_path
import sys
import copy
#NOTE: v01
# def filter_pred(preds):
#     cleaned_data = {}

#     for class_name, items in preds.items():
#         id_map = {}

#         for item in items:
#             item_id = item["id"]
#             if item_id not in id_map or not item['inside']:
#                 id_map[item_id] = item
#         cleaned_data[class_name] = list(id_map.values())
#     return cleaned_data
#v02: working
# def filter_pred(preds, press):
#     cleaned_data = {}

#     for class_name, items in preds.items():
#         id_map = {}
#         # Convert past data for the current class to a dictionary for quick lookup
#         past_states = {item["id"]: item["inside"] for item in press.get(class_name, [])}

#         for item in items:
#             item_id = item["id"]
#             current_state = item["inside"]
#             prev_state = past_states.get(item_id, False)  # Default to False if not present

#             # Determine the final state based on the previous and current states
#             if prev_state is True and current_state is False:
#                 final_state = False
#             elif prev_state is False and current_state is True:
#                 final_state = True
#             else:
#                 final_state = current_state

#             # Update the item with the final state
#             item["inside"] = final_state
#             id_map[item_id] = item

#         cleaned_data[class_name] = list(id_map.values())
#     return cleaned_data

def sort_data_keys(json_obj):
    """
    Sorts the keys within the "data" field of the input JSON object alphabetically.
    
    Parameters:
        json_obj (dict): A dictionary with a "data" key containing a dictionary.
        
    Returns:
        dict: A new dictionary with the "data" keys sorted alphabetically.
    """
    if "data" in json_obj and isinstance(json_obj["data"], dict):
        # Using OrderedDict to maintain sorted order explicitly.
        sorted_data = OrderedDict(sorted(json_obj["data"].items()))
        
        json_obj["data"] = sorted_data
    return json_obj

def find_repetitive_values(data):
    value_to_keys = {}  # Dictionary to map values to the keys they appear in
    repetitive_values = {}  # Store values that appear in multiple keys

    for key, values in data.items():
        for value in values:
            if value in value_to_keys:
                value_to_keys[value].append(key)
                repetitive_values[value] = value_to_keys[value]  # Store only if repeated
            else:
                value_to_keys[value] = [key]

    return repetitive_values if repetitive_values else None

def id_switch_adjustment(global_map, local_map):
    key_ids_global_map = {}
    key_ids_local_map = {}
    for key, values in global_map.items():
        x = []
        for i in values:
            x.append(i["id"])
        key_ids_global_map[key] = x
    for key, values in local_map.items():
        y = []
        for i in values:
            y.append(i["id"])
        key_ids_local_map[key] = y

    repeated_ids = find_repetitive_values(key_ids_global_map)
    if repeated_ids:
        print("//////////////////", repeated_ids)
        for key, values in repeated_ids.items():
            for value in values:
                try:
                    if key not in key_ids_local_map[value]:
                        global_map[value] = [item for item in global_map[value] if item["id"] != key]
                except:
                    pass
    print(global_map)
    return global_map



def filter_pred(preds, press):
    cleaned_data = {}

    for class_name, items in preds.items():
        id_map = {}
        # Convert past data for the current class to a dictionary for quick lookup
        past_states = {item["id"]: item["inside"] for item in press.get(class_name, [])}
        if not items:
            cleaned_data[class_name] = preds[class_name]
            continue
        for item in items:
            item_id = item["id"]
            current_state = item["inside"]

            # Initialize the state in id_map if not already present
            if item_id not in id_map:
                id_map[item_id] = {
                    "prev_state": past_states.get(item_id, False),  # Default to False
                    "current_states": set()
                }
            id_map[item_id]["current_states"].add(current_state)

        # Resolve final state based on conditions
        for item_id, state_info in id_map.items():
            prev_state = state_info["prev_state"]
            current_states = state_info["current_states"]

            if prev_state is True and current_states == {True, False}:
                final_state = False
            elif prev_state is False and current_states == {True, False}:
                final_state = True
            elif len(current_states) == 1:
                final_state = current_states.pop()
            else:
                final_state = False  # Default fallback for any unforeseen case

            # Add the resolved item to the cleaned data
            if class_name not in cleaned_data:
                cleaned_data[class_name] = []
            cleaned_data[class_name].append({"id": item_id, "inside": final_state})

    return cleaned_data


def simplyfy_dict(dict_item):
    mod_max_det = {}
    for class_name,items in dict_item.items():
        # print(class_name,items)
        mod_max_det_per_class = {}
        for i in items:
            id = i['id']
            status = i['inside']
            mod_max_det_per_class[id] = status
        # print('>>>>> ',mod_max_det_per_class)
        mod_max_det[class_name] = mod_max_det_per_class
    return mod_max_det
def configure_dict(dict_item):
    final_output = {}
    for i in dict_item:
        dict_list = []
        for j in dict_item[i]:
            dict_list.append({'id':j,'inside':dict_item[i][j]})
        final_output[i] = dict_list

    return final_output
        
def update_max_det(max_det,current_status):
    mod_max_det = simplyfy_dict(max_det)
    mod_current_status = simplyfy_dict(current_status)
    # Iterate through each category in current_status
    for category, status_items in mod_current_status.items():
        if category in mod_max_det:
            # Update the values in max_det based on current_status
            for key, value in status_items.items():
                if key in mod_max_det[category] and mod_max_det[category][key] != value:
                    mod_max_det[category][key] = value
    
    mod_max_det = configure_dict(mod_max_det)
    return mod_max_det

    
def draw_red_dot(image, bbox):
    """
    Draws a red dot at the center of the bounding box.
    
    Args:
        image: The input image.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
    """
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    cv2.circle(image, (int(center_x), int(center_y)), 20, (0, 0, 255), -1)  # Red dot
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (243, 150, 33), thickness=3)

    return image   
    


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
    line_points = [(800, 1280), (800, 1870), (1400, 1870), (1400, 1280)]#(500, 1680), (500, 1970), (710, 1970), (710, 1680)
    
    # Initialize ObjectCounter from ultralytics.solutions
    counter = solutions.ObjectCounter(
        view_img=view_img,
        reg_pts=line_points,
        classes_names=model.names,
        draw_tracks=False,
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
    global_map = {
        0: {
        # 'glove': [],
        # 'sponge': [],
        # 'woodspack': [],
        # 'gauze': [],
        'needle': [],
        # 'black_suture': [],
        # 'vesiloop': [],
        # 'needle_holder': [],
        # 'sucker': [],
        # 'clamp': [],
        'forceps': [],
        # 'scissors': [],
        # 'bovie': [],
        # 'scalpel': []
    },
    1: {
        # 'glove': [],
        # 'sponge': [],
        # 'woodspack': [],
        # 'gauze': [],
        'needle': [],
        # 'black_suture': [],
        # 'vesiloop': [],
        # 'needle_holder': [],
        # 'sucker': [],
        # 'clamp': [],
        'forceps': [],
        # 'scissors': [],
        # 'bovie': [],
        # 'scalpel': []
        }
    }#{0:{'glove':[], 'sponge':[], 'woodspack':[], 'gauze':[], 'needle':[]}, 1:{'glove':[], 'sponge':[], 'woodspack':[], 'gauze':[], 'needle':[]}}  # To store global IDs
    max_det_ = {}
    # max_in_det = { "sponge":0, "woodspack":0, "gauze":0, 
    #               "needle":0,   
    #               "needle_holder":0, "forceps":0, 
    #              }
    max_in_det = {"gauze":0, 
                   
                  "needle":0, "forceps":0, 
                 }
    # class_names = ["sponge", "woodspack","gauze", 
    #                "needle", "needle_holder", "forceps", 
    #                ]
    class_names = ["gauze", 
                   "needle", "forceps", 
                   ]
    while cap.isOpened():
        
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has completed.")
            break
        # print(f">>>>>>>>>global_map: {global_map}")

        # print(f"size of an image is: {sys.getsizeof(im0)}")

        

        # Perform object tracking
        tracks = model.track(im0, persist=True, conf=0.05, iou=0.005, show=False, imgsz=2496, tracker="botsort.yaml", classes=[6,14])#[1,5,11,13,9]
        im, im_status = counter.start_counting(im0, tracks )

        im = cv2.rectangle(im, (500,900), (1600,2000), (0,255,255), 5) ###(600,900), (1400,2000), (0,0,255)

        i += 1
        if not im_status or len(im_status[0]) == 0:
            print(f"No objects detected or tracked in frame {i}.")
            #NOTE: if i > 1 else {} not necessary because initalize with 2 frames at beginning
            global_map[i] = global_map[i - 1] #if i > 1 else {}
            cv2.imwrite(f"/root/ws/ultralytics/gui_data/frame_{i}.jpg", im)
            with open(f"/root/ws/ultralytics/gui_data/frame_{i}.json", "w") as f:
                json.dump(sort_data_keys({"data":global_map[i]}), f, indent=4)
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
            # if is_inside:
                # im = draw_red_dot(im, bbox)
            val = {"id": obj_id, "inside": is_inside}
            # Initialize the list for the class_name key if it doesn't exist
            if class_name not in local_map:
                local_map[class_name] = []
            
            # Append the value to the class_name list
            local_map[class_name].append(val)
            # print(">>>>>>>>>>>>>.local_map>>>>>>", local_map)
        local_map_ = copy.deepcopy(local_map)
        global_map[i] = local_map #{"id": obj_id,  "inside": is_inside}#"class": class_name, "bbox": bbox,
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",i)
        for class_name, item_list in global_map[i].items():
            #handle not detected classes
            

            total_items = len(item_list)  # Total number of items
            inside_true_count = sum(1 for item in item_list if item['inside'])  # Count where 'inside' is True #```if item['inside']```
            inside_false_count = sum(1 for item in item_list if not item['inside'])
            #NOTE changes: inside_true_count >1 to inside_true_count >=1
            # print(max_in_det)
            if inside_true_count >1 and inside_true_count>= max_in_det[class_name]:
                
                max_det = item_list#{class_name: item_list}#global_map[i]
                max_det_[class_name] = max_det
                # print(f"class_name: {class_name}, max_det: {max_det_[class_name]}")
                max_in_det[class_name] = inside_true_count
            else:
                # Default behavior when condition is not satisfied
                max_det = {class_name: [] for class_name in global_map[i].keys()}  # Initialize empty for each class

            # print(f"Frame {i}: {max_det_}")  
            # print(">>>>>",global_map.values())
            #NOTE: Append the max_det to the global_map according to different classes
            # max_det_[class_name] = filter_max_det(global_map.values()[class_name][i-30:i], global_map.values()[class_name][i], max_det_[class_name])

            max_det_ = update_max_det(max_det_, global_map[i])
            #Handle non-detected items 
            
            # print(">>>>>>>>Max_det>>>>", max_det_)
            if max_det_.get(class_name): 
                # print(">>>>>>>>>",class_name, max_det_[class_name])
                # print("<<<<<<<<<<<", global_map[i][class_name])
                # global_map[i][class_name].extend(max_det_[class_name])
                existing_items = global_map[i][class_name]
                new_items = [item for item in max_det_[class_name] if item not in existing_items]
                existing_items = set(tuple(item.items()) for item in global_map[i][class_name])  # Convert dicts to hashable tuples
                new_items = [item for item in max_det_[class_name] if tuple(item.items()) not in existing_items]
                if new_items:
                    # print(f"$$$$$$$$$$$$$$$$$$$$$$ Frame: {i}, Class: {class_name} with: {new_items}")
                    global_map[i][class_name].extend(new_items)
                # global_map[i][class_name].append(max_det_[class_name][-1])
        
        # keys_to_check = [
        #                 "sponge", "woodspack","gauze", 
        #                 "needle", "needle_holder", "forceps",
        #             ]
        keys_to_check = [
                        # "gauze", 
                        "needle", "forceps",
                    ]
        

        for key in keys_to_check:
            if key not in global_map[i] and i >= 1:
                global_map[i][key] = global_map[i - 1][key]

        # print("Before:",global_map[i])
        global_map[i] = filter_pred(global_map[i], global_map[i-1])
        # print("After:", global_map[i])
            # except:
            #     print("going through pass***********************>>>>>>>")
            #     pass
                # Find the difference
                # existing_items = global_map[i][class_name]
                # new_items = [item for item in max_det_[class_name] if item not in existing_items]
                
                # if new_items:
                #     print(f"$$$$$$$$$$$$$$$$$$$$$$ Frame: {i}, Class: {class_name} with : {new_items}")
                #     global_map[i][class_name].extend(new_items)  # Append only the new items
            # if max_det.get(class_name):  # Ensure `class_name` exists in `max_det`
            #     global_map[i][class_name].append(list(max_det[class_name])[-1] if max_det[class_name] else None)
        with open(f"/root/ws/ultralytics/gui_data/frame_{i}.json", "w") as f:
            json.dump(sort_data_keys({"data":global_map[i]}), f, indent=4)

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
