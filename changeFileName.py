"""
File: changeFileName.py
Description: A script to batch rename image files in the dataset directory,
             specifically changing filenames from 'norain-*.png' to 'rain-*.png'
Author: Yiwei Gui
Date: April 5, 2025
Usage: python changeFileName.py
"""

import os

# Set the target directory path
target_dir = r"/scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/Deep-Fourier-Upsampling/Dataset/RainTrainH_modified/RainTrainH_modified/norain"  

# Iterate through all files in the directory
for filename in os.listdir(target_dir):
    if filename.startswith("norain-") and filename.endswith(".png"):
        new_name = filename.replace("norain-", "rain-", 1)
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

print("All filenames have been successfully replaced!")
