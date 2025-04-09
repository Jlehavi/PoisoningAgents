#!/usr/bin/env python
"""
Script to perform a fresh capture of 120 images (40 per labeler) and distribute them properly.
"""

import os
import json
import shutil
import time
import random
from data_capture_and_label import IntersectionDataCapture

def clean_directory(directory):
    """Remove all PNG files from a directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Removed: {file_path}")
    else:
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def fresh_capture_and_distribute(target_count=120, labelers=None):
    """Capture new data and distribute it properly."""
    if labelers is None:
        labelers = ["Sean", "Simon", "Jack"]
    
    # Create data capture instance
    data_capture = IntersectionDataCapture()
    
    # Clean the image directories
    print("Cleaning image directories...")
    clean_directory(data_capture.image_dir)
    
    for labeler in labelers:
        labeler_dir = os.path.join(data_capture.data_dir, f"labeler_{labeler}")
        labeler_image_dir = os.path.join(labeler_dir, "images")
        clean_directory(labeler_image_dir)
    
    # Clear existing scene labels
    data_capture.scene_labels = []
    
    # Calculate how many episodes to run to get approximately the target count
    # Estimate about 1-2 images per episode to be conservative
    avg_images_per_episode = 1
    episodes_needed = max(20, target_count // avg_images_per_episode)
    max_episodes = 200  # Limit to prevent running forever
    
    print(f"Starting capture of {target_count} images...")
    print(f"Will run up to {episodes_needed} episodes")
    
    # Capture data in batches
    total_captures = 0
    episodes_run = 0
    
    while total_captures < target_count and episodes_run < max_episodes:
        # Capture in smaller batches for more frequent updates
        batch_size = min(10, max_episodes - episodes_run)
        print(f"Running {batch_size} episodes (total {episodes_run}/{max_episodes})...")
        
        # Capture data for a batch of episodes
        pre_count = len(data_capture.scene_labels)
        data_capture.capture_data(
            num_episodes=batch_size,
            num_steps_per_episode=30,
            auto_label=True,
            manual_label=False
        )
        
        # Update counts
        post_count = len(data_capture.scene_labels)
        new_captures = post_count - pre_count
        total_captures = post_count
        episodes_run += batch_size
        
        print(f"Captured {new_captures} new images in this batch")
        print(f"Total captures so far: {total_captures}/{target_count}")
        
        # Save progress after each batch
        data_capture.save_labels()
        
        # If we're getting close to the target, adjust batch size
        if total_captures >= target_count * 0.9:
            batch_size = 1
    
    print(f"Finished capture with {total_captures} images")
    
    # Trim excess images if needed
    if total_captures > target_count:
        print(f"Trimming excess images ({total_captures} -> {target_count})")
        data_capture.scene_labels = data_capture.scene_labels[:target_count]
        data_capture.save_labels()
    
    # Distribute images to labelers
    images_per_labeler = target_count // len(labelers)
    print(f"Distributing {target_count} images to {len(labelers)} labelers")
    print(f"Each labeler will get {images_per_labeler} images")
    
    # Shuffle the labels for random distribution
    random.shuffle(data_capture.scene_labels)
    
    # Prepare labeler directories and assignments
    labeler_assignments = {}
    for labeler in labelers:
        labeler_dir = os.path.join(data_capture.data_dir, f"labeler_{labeler}")
        labeler_image_dir = os.path.join(labeler_dir, "images")
        
        if not os.path.exists(labeler_dir):
            os.makedirs(labeler_dir)
        if not os.path.exists(labeler_image_dir):
            os.makedirs(labeler_image_dir)
        
        labeler_assignments[labeler] = []
    
    # Distribute the images
    for i, label in enumerate(data_capture.scene_labels):
        labeler_idx = i // images_per_labeler
        if labeler_idx >= len(labelers):
            break
        
        labeler = labelers[labeler_idx]
        labeler_assignments[labeler].append(label)
        
        # Copy the image file
        src_path = os.path.join(data_capture.image_dir, label["filename"])
        dst_path = os.path.join(data_capture.data_dir, f"labeler_{labeler}/images", label["filename"])
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"WARNING: Source image not found: {src_path}")
    
    # Save the assignments
    for labeler, assignments in labeler_assignments.items():
        labeler_file = os.path.join(data_capture.data_dir, f"labeler_{labeler}/scene_labels.json")
        
        with open(labeler_file, "w") as f:
            json.dump(assignments, f, indent=2)
        
        print(f"Assigned {len(assignments)} images to {labeler}")
        
        # Verify the image files
        expected_files = [label["filename"] for label in assignments]
        actual_files = os.listdir(os.path.join(data_capture.data_dir, f"labeler_{labeler}/images"))
        actual_pngs = [f for f in actual_files if f.endswith(".png")]
        
        print(f"  Expected: {len(expected_files)} images")
        print(f"  Actual: {len(actual_pngs)} images")
        
        missing = [f for f in expected_files if f not in actual_files]
        if missing:
            print(f"  Missing {len(missing)} images")
    
    print("\nDistribution complete!")
    print("Instructions for labelers:")
    for labeler in labelers:
        print(f"\n{labeler}:")
        print(f"1. Your images are in: intersection_data_captures/labeler_{labeler}/images/")
        print(f"2. Your label file is: intersection_data_captures/labeler_{labeler}/scene_labels.json")
        print(f"3. Please review the images and modify the 'best_move' field if needed")

if __name__ == "__main__":
    fresh_capture_and_distribute() 