#!/usr/bin/env python
"""
Script to copy all images that are referenced in the scene_labels.json files.
This will ensure all labelers have access to the images they're supposed to label.
"""

import os
import json
import glob
import shutil
import re

def copy_all_referenced_images(data_dir="intersection_data_captures", labelers=None):
    """Copy all images referenced in scene_labels.json files."""
    if labelers is None:
        labelers = ["Sean", "Simon", "Jack"]
    
    # Main images directory
    main_image_dir = os.path.join(data_dir, "images")
    
    if not os.path.exists(main_image_dir):
        print(f"Main image directory not found: {main_image_dir}")
        return
    
    # First, create a mapping of all available images (both timestamped and not)
    available_images = {}
    
    # Get all image files
    image_files = glob.glob(os.path.join(main_image_dir, "*.png"))
    
    print(f"Found {len(image_files)} images in the main directory")
    
    # Process each image file and build a mapping of episode/step to best available file
    for image_path in image_files:
        filename = os.path.basename(image_path)
        
        # Extract episode and step using regex - handle both timestamped and non-timestamped files
        match = re.match(r"car_scene_ep(\d+)_step(\d+)(?:_\d+)?\.png", filename)
        if match:
            episode = int(match.group(1))
            step = int(match.group(2))
            key = (episode, step)
            
            # Check if this is a newer file than what we already have
            if key not in available_images or filename > available_images[key]["filename"]:
                available_images[key] = {
                    "filename": filename,
                    "path": image_path
                }
    
    print(f"Found {len(available_images)} unique episode/step combinations")
    
    # For each labeler, process their scene_labels.json
    for labeler in labelers:
        labeler_dir = os.path.join(data_dir, f"labeler_{labeler}")
        labeler_image_dir = os.path.join(labeler_dir, "images")
        labeler_json = os.path.join(labeler_dir, "scene_labels.json")
        
        if not os.path.exists(labeler_json):
            print(f"No labels found for {labeler} at {labeler_json}")
            continue
        
        # Ensure image directory exists
        if not os.path.exists(labeler_image_dir):
            os.makedirs(labeler_image_dir)
        
        # Load the label file
        with open(labeler_json, "r") as f:
            labels = json.load(f)
        
        # Process each label
        missing_count = 0
        copy_count = 0
        unchanged_count = 0
        updated_count = 0
        
        for i, label in enumerate(labels):
            # Get the current filename
            old_filename = label["filename"]
            
            # Extract episode and step
            match = re.match(r"car_scene_ep(\d+)_step(\d+)(?:_\d+)?\.png", old_filename)
            
            if match:
                episode = int(match.group(1))
                step = int(match.group(2))
                key = (episode, step)
                
                # If we have this episode/step in our available images
                if key in available_images:
                    best_filename = available_images[key]["filename"]
                    
                    # If the filename in the label is different, update it
                    if old_filename != best_filename:
                        label["filename"] = best_filename
                        updated_count += 1
                    else:
                        unchanged_count += 1
                    
                    # Copy the image to the labeler's directory if not already there
                    dst_path = os.path.join(labeler_image_dir, best_filename)
                    src_path = available_images[key]["path"]
                    
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        copy_count += 1
                else:
                    missing_count += 1
                    print(f"WARNING: No image found for ep{episode}_step{step} referenced by {labeler}")
            else:
                missing_count += 1
                print(f"WARNING: Could not parse filename: {old_filename}")
        
        # Save the updated labels
        with open(labeler_json, "w") as f:
            json.dump(labels, f, indent=2)
        
        # Count how many images the labeler actually has now
        actual_images = len(glob.glob(os.path.join(labeler_image_dir, "*.png")))
        
        print(f"\nResults for {labeler}:")
        print(f"  Total assigned images: {len(labels)}")
        print(f"  Updated references: {updated_count}")
        print(f"  Unchanged references: {unchanged_count}")
        print(f"  Missing images: {missing_count}")
        print(f"  Images copied: {copy_count}")
        print(f"  Actual images in directory: {actual_images}")

if __name__ == "__main__":
    copy_all_referenced_images() 