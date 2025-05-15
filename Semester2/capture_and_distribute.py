#!/usr/bin/env python
"""
Script to capture intersection data and distribute it to labelers.
This will:
1. Capture 120 images from the intersection environment
2. Auto-label all images
3. Distribute the images among Sean, Simon, and Jack
"""

import os
import argparse
from data_capture_and_label import IntersectionDataCapture

def main():
    """Main function to capture data and distribute to labelers."""
    parser = argparse.ArgumentParser(description='Capture intersection data and distribute to labelers.')
    parser.add_argument('--no-capture', action='store_true',
                       help='Skip capture and only distribute existing data')
    parser.add_argument('--target-count', type=int, default=120,
                       help='Target number of images to capture')
    parser.add_argument('--labelers', nargs='+', default=["Sean", "Simon", "Jack"],
                       help='Names of labelers to distribute tasks to')
    parser.add_argument('--max-episodes', type=int, default=200,
                       help='Maximum number of episodes to run')
    args = parser.parse_args()
    
    # Create data capture instance
    data_capture = IntersectionDataCapture()
    
    # Check if we should load existing data
    has_existing_data = False
    if os.path.exists(os.path.join(data_capture.data_dir, "scene_labels.json")):
        print("Loading existing scene labels...")
        data_capture.load_labels()
        has_existing_data = True
        print(f"Currently have {len(data_capture.scene_labels)} labeled images")
    
    # Capture data if needed
    if not args.no_capture:
        # Calculate episodes needed - using a lower estimate to ensure we run enough episodes
        # The actual capture rate can vary significantly
        avg_images_per_episode = 1  # Lower this estimate to run more episodes
        episodes_needed = max(10, args.target_count // avg_images_per_episode)
        
        # Cap at max-episodes to prevent running too long
        episodes_needed = min(episodes_needed, args.max_episodes)
        
        print(f"Starting capture of approximately {args.target_count} images...")
        print(f"Will run up to {episodes_needed} episodes")
        
        # Current image count
        current_count = len(data_capture.scene_labels) if has_existing_data else 0
        
        # Capture in batches
        total_captures = current_count
        episodes_run = 0
        
        while total_captures < args.target_count and episodes_run < episodes_needed:
            # Capture in smaller batches for more frequent updates
            batch_size = min(10, episodes_needed - episodes_run)
            print(f"Running {batch_size} episodes (total {episodes_run}/{episodes_needed})...")
            
            # Get the current count of labels
            pre_count = len(data_capture.scene_labels)
            
            # Capture data for a batch of episodes
            data_capture.capture_data(
                num_episodes=batch_size, 
                num_steps_per_episode=30,  # Increase steps per episode
                auto_label=True, 
                manual_label=False
            )
            
            # Update counts
            post_count = len(data_capture.scene_labels)
            new_captures = post_count - pre_count
            total_captures = post_count
            episodes_run += batch_size
            
            print(f"Captured {new_captures} new images in this batch")
            print(f"Total captures so far: {total_captures}/{args.target_count}")
            
            # Save progress after each batch
            data_capture.save_labels()
            
            # If we're getting very few captures, consider breaking early
            if episodes_run > 50 and total_captures < args.target_count / 4:
                print("WARNING: Low capture rate detected. Consider adjusting parameters.")
                
            # If we're getting close to the target, run individual episodes to avoid overshooting too much
            if total_captures >= args.target_count * 0.9:
                batch_size = 1
    elif not has_existing_data:
        print("No existing data found and --no-capture specified. Please run without --no-capture first.")
        return
    
    # Distribute the data to labelers
    print(f"Distributing data to labelers: {', '.join(args.labelers)}")
    data_capture.distribute_labeling_tasks(args.labelers)
    
    # Print instructions for labelers
    print("\n" + "=" * 50)
    print("INSTRUCTIONS FOR LABELERS")
    print("=" * 50)
    for labeler in args.labelers:
        print(f"\nInstructions for {labeler}:")
        print(f"1. Your images are in: intersection_data_captures/labeler_{labeler}/images/")
        print(f"2. Your label file is: intersection_data_captures/labeler_{labeler}/scene_labels.json")
        print(f"3. Please review the images and modify the 'best_move' field in the label file if needed")
        print(f"4. Choose from these options:")
        
        # Print action mapping
        action_map = data_capture.define_action_mapping()
        for idx, action_name in action_map.items():
            print(f"   - {idx}: {action_name}")
    
    print("\nAfter all labelers have completed their tasks, run:")
    print("python data_capture_and_label.py --merge")
    print("=" * 50)

if __name__ == "__main__":
    main() 