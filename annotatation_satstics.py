#!/usr/bin/env python3
#@Author: Akash Manna
#@Date:14/05/2023

import os
import json
import matplotlib.pyplot as plt
import shutil
import argparse

def image_statistics_generator(base_dir, needle_dest_path, save_needle_images):
    category_counts = {}

    corrupt_file = []
    # Loop through the JSON files in the directory
    counter= 0
    for json_file in [f for f in os.listdir(base_dir) if f.endswith('.json')]: #os.listdir(base_dir):
        val_path= base_dir +json_file.split(".")[0] + ".jpg"#".jpg" 
        if os.path.exists(val_path):
            print("[Manna-log]: Processing: ", json_file)
            try:
                if json_file.endswith('.json'):
                    # Load the JSON file as a Python dict
                    with open(os.path.join(base_dir, json_file)) as f:
                        data = json.load(f)

                    # Count the number of annotations in the file for each category
                    for annotation in data['shapes']:
                        category = annotation['label']
                        if category not in category_counts:
                            category_counts[category] = 0
                        category_counts[category] += 1
                        ##NOTE: add needle class
                        if False: #save_needle_images== False:
                            if category=="needle":
                                # print(">>>>>", category)
                                # print(">>>>>", needle_dest_path+ val_path.split("/")[-1])
                                # print(">>>>val_path", val_path)
                                shutil.copy(val_path, needle_dest_path+ val_path.split("/")[-1])
                                shutil.copy(os.path.join(base_dir, json_file), needle_dest_path+"/"+ json_file)
            except:
                corrupt_file.append(json_file)
            counter+=1
            print(f"Images analysed: {counter}")
    print("[Manna-log]: List of corrpupted files is: ", len(corrupt_file))

    # Plot the results
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.bar(categories, counts)

    # Add text annotations to the bars
    ax.bar_label(ax.containers[-1], label_type='edge')
    print(">>>>>>>>>>>>>>", categories)
    ax.set_xlabel('Category')
    ax.set_ylabel('# annotations')
    ax.xaxis.set_ticks(categories)
    ax.set_xticklabels(categories, rotation=90)
    # Show the plot
    plt.savefig("all_dataset_stat_070325.png")

if __name__ =="__main__":


    # # Specify the directory containing the Labelme annotated JSON files
    # base_dir = '/media/binarisoftware/HDD/shared/data_29_05_2023/'
    # # Create a dictionary to hold the category counts
    
    # needle_dest_path= "/media/binarisoftware/HDD/shared/data_29_05_2023/needle_images/"

    parser = argparse.ArgumentParser(description='argument for data analysis')
    parser.add_argument('--base_dir', default= "", type=str, help='Description of your argument')
    parser.add_argument("--save_needle_images", action='store_true')
    parser.add_argument('--needle_image_path', type=str, default= "." ,help='Needle image path')
    args = parser.parse_args()
    print()
    print("Arguments used are: ", args)

    opt= args

    if os.path.exists(opt.needle_image_path):
        print('The directory exists, over-writting!!!')
    else:
        print('The directory does not exist, creating the directory')
        os.makedirs(opt.needle_image_path)

    image_statistics_generator(base_dir= opt.base_dir, 
                               needle_dest_path= opt.needle_image_path,
                               save_needle_images= opt.save_needle_images)

