import os
import json
import glob

# Directory containing JSON files
directory_path = '/mnt/data/dataset/'

# Function to update the label in a single annotation file
def update_label(annotation):
    for shape in annotation['shapes']:
        if shape['label'] == 'needle?':
            shape['label'] = 'needle'
        if shape['label'] == 'needle ?':
            shape['label'] = 'needle'
        if shape['label'] == 'woods pack':
            shape['label'] = 'woodspack'
        if shape['label'] == 'woods_pack':
            shape['label'] = 'woodspack'
        if shape['label'] == 'scissor':
            shape['label'] = 'scissors'
        if shape['label'] == 'black suture':
            shape['label'] = 'black_suture'
        if shape['label'] == 'needle holder':
            shape['label'] = 'needle_holder'
        if shape['label'] == 'clamp 4':
            shape['label'] = 'clamp'
        if shape['label'] == 'clamp 7':
            shape['label'] = 'clamp'
        if shape['label'] == 'clamp 5':
            shape['label'] = 'clamp'
        if shape['label'] == 'clamp 6':
            shape['label'] = 'clamp'
        if shape['label'] == 'clamp 9':
            shape['label'] = 'clamp'
        if shape['label'] == 'clamp 8':
            shape['label'] = 'clamp'
        if shape['label'] == 'wire cutter clamp':
            shape['label'] = 'clamp'
        if shape['label'] == 'wire cutter':
            shape['label'] = 'clamp'
        if shape['label'] == 'hocky_puck':
            shape['label'] = 'hockey_puck'
        if shape['label'] == 'umbilical tape':
            shape['label'] = 'umbilical_tape'
        if shape['label'] == 'clamp_rubber__shod':
            shape['label'] = 'clamp_rubber_shod'
        if shape['label'] == 'hegar dilator':
            shape['label'] = 'hegar_dilator'
    return annotation

# Process all JSON files in the directory
for file_path in glob.glob(os.path.join(directory_path, '*.json')):
    print(f"Updating {file_path}")
    with open(file_path, 'r') as file:
        annotation_content = json.load(file)
        
    updated_annotation = update_label(annotation_content)
    
    with open(file_path, 'w') as file:
        json.dump(updated_annotation, file, indent=4)

    print(f"Updated {file_path}")

# Example directory path usage:
# directory_path = '/path/to/your/json/files'

