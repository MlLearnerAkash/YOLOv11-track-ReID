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


def filter_pred(preds, press):
    cleaned_data = {}

    for class_name, items in preds.items():
        
        id_map = {}
        # Convert past data for the current class to a dictionary for quick lookup
        past_states = {item["id"]: item["inside"] for item in press.get(class_name, [])}

        for item in items:
            item_id = item["id"]
            current_state = item["inside"]
            prev_state = past_states.get(item_id, False)  # Default to False if not present

            # Determine the final state based on the previous and current states
            if prev_state is True and current_state is False:
                final_state = False
            elif prev_state is False and current_state is True:
                final_state = True
            else:
                final_state = current_state

            # Update the item with the final state
            item["inside"] = final_state
            id_map[item_id] = item

        cleaned_data[class_name] = list(id_map.values())
    return cleaned_data

def filter_pred(preds, press):
    cleaned_data = {}

    for class_name, items in preds.items():
        id_map = {}
        # Convert past data for the current class to a dictionary for quick lookup
        past_states = {item["id"]: item["inside"] for item in press.get(class_name, [])}

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


# def update_max_det(max_det_, current_status):
#     """
#     Updates the `inside` field in `max_det_` based on the corresponding values in `current_status`.

#     Args:
#         max_det_ (dict): Reference dictionary, structured by categories (e.g., 'glove', 'sponge').
#         current_status (dict): Current status dictionary, structured similarly to `max_det_`.

#     Returns:
#         dict: Updated `max_det_` with `inside` fields modified based on `current_status`.
#     """
#     # Iterate through each category in current_status
#     for category, status_items in current_status.items():
#         if category in max_det:
#             # Create a lookup for `id` in current_status for quick access
#             status_lookup = {item['id']: item['inside'] for item in status_items}
            
#             # Update `max_det` for matching ids
#             for max_item in max_det[category]:
#                 if max_item['id'] in status_lookup:
#                     max_item['inside'] = status_lookup[max_item['id']]
    
#     return max_det
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




if __name__ == "__main__":
    test_current = {'glove': [{'id': 1, 'inside': True}, {'id': 2, 'inside': False}, {'id': 3, 'inside': True}, {'id': 10, 'inside': False}, {'id': 11, 'inside': False}, {'id': 11, 'inside': True}], 'sponge': []}
    test_past = {'glove': [{'id': 1, 'inside': True}, {'id': 2, 'inside': False}, {'id': 3, 'inside': True}, {'id': 10, 'inside': False}, {'id': 11, 'inside': False}], 'sponge': []}

    # print(filter_pred(test_current, test_past))

    # current_status= {'glove': [{'id': 1, 'inside': False}, {'id': 2, 'inside': True}]}
    # max_det = {'glove': [{'id': 1, 'inside': True}, {'id': 2, 'inside': False}]}

    current_status = {'glove': [{'id': 1, 'inside': True}, {'id': 2, 'inside': False}, {'id': 3, 'inside': True}, {'id': 11, 'inside': False}], 'sponge': []}
    max_det = {'glove': [{'id': 1, 'inside': False}, {'id': 2, 'inside': False}, {'id': 3, 'inside': True}, {'id': 10, 'inside': False}, {'id': 11, 'inside': False}], 'sponge': [{'id': 10, 'inside': False}, {'id': 11, 'inside': False}]}
    # print(simplyfy_dict(current_status))

   
    print(update_max_det(max_det,current_status))