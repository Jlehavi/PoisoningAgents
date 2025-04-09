#!/usr/bin/env python
"""
Script to copy missing images from the main directory to each labeler's directory.
This will ensure all images referenced in scene_labels.json are present in the images directory.
"""

import os
import json
import shutil

def copy_missing_images(data_dir="intersection_data_captures", labelers=None):
    """Copy missing images for each labeler."""
    if labelers is None:
        labelers = ["Sean", "Simon", "Jack"]
    
    # Main images directory
    main_image_dir = os.path.join(data_dir, "images")
    
    if not os.path.exists(main_image_dir):
        print(f"Main image directory not found: {main_image_dir}")
        return
    
    for labeler in labelers:
        # Labeler directories
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
        
        # Get filenames from labels
        expected_files = [label["filename"] for label in labels]
        
        # Check which files exist
        existing_files = set(os.listdir(labeler_image_dir))
        
        # Copy missing files
        missing_count = 0
        for filename in expected_files:
            if filename not in existing_files:
                src_path = os.path.join(main_image_dir, filename)
                dst_path = os.path.join(labeler_image_dir, filename)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    missing_count += 1
                else:
                    print(f"WARNING: Source file not found: {src_path}")
        
        print(f"Copied {missing_count} missing images for {labeler}")
        print(f"Total images for {labeler}: {len(expected_files)}")

if __name__ == "__main__":
    copy_missing_images() 