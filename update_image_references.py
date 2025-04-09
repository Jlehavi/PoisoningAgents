#!/usr/bin/env python
"""
Script to update the filenames in scene_labels.json for each labeler with the latest timestamped filenames.
This will fix the discrepancy between the filenames in the JSON and the actual image files.
"""

import os
import json
import glob
import shutil
import re

def update_image_references(data_dir="intersection_data_captures", labelers=None):
    """Update image references in scene_labels.json files."""
    if labelers is None:
        labelers = ["Sean", "Simon", "Jack"]
    
    # Main images directory
    main_image_dir = os.path.join(data_dir, "images")
    
    if not os.path.exists(main_image_dir):
        print(f"Main image directory not found: {main_image_dir}")
        return
    
    # Get a mapping of episode and step to the latest timestamped image
    latest_images = {}
    
    # Get all image files
    image_files = glob.glob(os.path.join(main_image_dir, "*.png"))
    
    # Process each image file
    for image_path in image_files:
        filename = os.path.basename(image_path)
        
        # Extract episode and step using regex - handle both timestamped and non-timestamped files
        match = re.match(r"car_scene_ep(\d+)_step(\d+)(?:_\d+)?\.png", filename)
        if match:
            episode = int(match.group(1))
            step = int(match.group(2))
            key = (episode, step)
            
            # Check if this is a newer file than what we already have
            if key not in latest_images or filename > latest_images[key]["filename"]:
                latest_images[key] = {
                    "filename": filename,
                    "path": image_path
                }
    
    print(f"Found {len(latest_images)} unique episode/step combinations")
    
    # Update each labeler's labels and copy the images
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
        
        # Update image references and copy images
        updates = 0
        for label in labels:
            # Extract episode and step from the current filename
            match = re.match(r"car_scene_ep(\d+)_step(\d+)(?:_\d+)?\.png", label["filename"])
            if match:
                episode = int(match.group(1))
                step = int(match.group(2))
                key = (episode, step)
                
                # Check if we have a newer timestamped file for this episode/step
                if key in latest_images:
                    # Update the filename
                    old_filename = label["filename"]
                    new_filename = latest_images[key]["filename"]
                    
                    if old_filename != new_filename:
                        label["filename"] = new_filename
                        updates += 1
                        
                        # Copy the new image to the labeler's directory
                        dst_path = os.path.join(labeler_image_dir, new_filename)
                        src_path = latest_images[key]["path"]
                        
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
                            
                            # Remove the old file if it exists
                            old_path = os.path.join(labeler_image_dir, old_filename)
                            if os.path.exists(old_path):
                                os.remove(old_path)
        
        # Save the updated labels
        with open(labeler_json, "w") as f:
            json.dump(labels, f, indent=2)
        
        print(f"Updated {updates} image references for {labeler}")
        print(f"Total images for {labeler}: {len(labels)}")
        
        # Count the actual image files in the directory
        image_count = len(glob.glob(os.path.join(labeler_image_dir, "*.png")))
        print(f"Actual image files in directory: {image_count}")

if __name__ == "__main__":
    update_image_references() 