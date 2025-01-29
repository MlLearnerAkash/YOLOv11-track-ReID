import os
import shutil

def copy_and_clean_files(source_dir, destination_dir):
    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(source_file):
            base_name, ext = os.path.splitext(filename)
            destination_file = os.path.join(destination_dir, filename)
            
            # Remove files in the destination directory with the same base name but with a suffix
            for dest_file in os.listdir(destination_dir):
                if dest_file.startswith(base_name + "_") and dest_file.endswith(ext):
                    os.remove(os.path.join(destination_dir, dest_file))
                    print(f"Removed: {dest_file}")
            
            # Copy the file from source to destination
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {filename}")

# Example usage
source_dir = "/mnt/data/3000_images_needle"
destination_dir = "/mnt/data/dataset"
copy_and_clean_files(source_dir, destination_dir)