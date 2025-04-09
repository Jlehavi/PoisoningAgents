import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import gymnasium
import highway_env
from typing import Dict, List, Tuple, Any, Optional
import argparse

class IntersectionDataCapture:
    """
    Class for capturing and labeling intersection environment data.
    """
    def __init__(self, data_dir: str = "intersection_data_captures"):
        """Initialize the data capture with the output directory."""
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.scene_labels = []
        
        # Create directories if they don't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
    
    def setup_environment(self):
        """Create and configure the intersection environment."""
        env = gymnasium.make(
            "intersection-v0",
            render_mode="rgb_array",
            config={
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True
                },
                "duration": 13,  # [s]
                "destination": "o1",
                "initial_vehicle_count": 15,  # Increased from 10
                "spawn_probability": 0.8,     # Increased from 0.6
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
            }
        )
        return env
    
    def is_at_intersection_entrance(self, obs: np.ndarray) -> bool:
        """
        Determine if the ego vehicle is close to the intersection entrance with other cars present.
        
        Args:
            obs: Current observation
            
        Returns:
            bool: True if the vehicle is close to the intersection with other cars
        """
        # Check if ego vehicle is present
        if obs[0][0] <= 0.5:
            return False
        
        # For the intersection-v0 environment, the y-coordinate indicates the position
        # Smaller y values mean closer to the intersection
        ego_y = obs[0][2]
        
        # Count other vehicles in the scene
        other_vehicles_count = sum(1 for v in obs[1:] if v[0] > 0.5)
        
        # When y is between 0.1 and 0.3, the vehicle is much closer to the intersection
        # We also require at least 2 other vehicles to be present for a proper decision
        return 0.1 <= ego_y <= 0.3 and other_vehicles_count >= 2
    
    def capture_frame(self, env, obs, step_num: int, episode_num: int) -> str:
        """
        Render and save a frame with velocity information overlay.
        
        Args:
            env: The environment instance
            obs: The current observation
            step_num: Current step number
            episode_num: Current episode number
            
        Returns:
            str: The filename of the saved image
        """
        # Get RGB array from environment
        frame = env.render()
        
        # Convert to PIL Image for adding overlays
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Try to find a font that works across platforms
        try:
            font = ImageFont.truetype("Arial", 12)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans", 12)
            except IOError:
                font = ImageFont.load_default()
        
        # Add velocity and position info for all vehicles
        y_offset = 10  # Starting y position for text
        
        # First, collect all visible vehicles and sort them by x position from left to right
        visible_vehicles = []
        for i, vehicle in enumerate(obs):
            if vehicle[0] > 0.5:  # If vehicle is present
                # Get vehicle position in observation
                veh_x, veh_y = vehicle[1], vehicle[2]
                vx, vy = vehicle[3], vehicle[4]
                velocity = np.sqrt(vx**2 + vy**2)
                
                # Scale factors to map observation coordinates to screen coordinates
                screen_width, screen_height = 600, 600
                x_scale = screen_width / 2
                y_scale = screen_height / 2
                
                # Map vehicle position to screen coordinates
                screen_x = int(screen_width/2 + veh_x * x_scale)
                screen_y = int(screen_height/2 - veh_y * y_scale)  # Invert y-axis
                
                visible_vehicles.append({
                    'id': i,
                    'is_ego': i == 0,
                    'x_pos': veh_x,  # Original x position for sorting
                    'y_pos': veh_y,  # Original y position
                    'screen_x': screen_x,
                    'screen_y': screen_y,
                    'vx': vx,
                    'vy': vy,
                    'velocity': velocity
                })
        
        # Sort vehicles by x position (left to right)
        sorted_vehicles = sorted(visible_vehicles, key=lambda v: v['x_pos'])
        
        # Reassign IDs based on x position (left to right)
        for idx, vehicle in enumerate(sorted_vehicles):
            vehicle['display_id'] = idx
            
            # Create vehicle label for the info panel
            vehicle_label = "Ego Vehicle" if vehicle['is_ego'] else f"Vehicle {vehicle['display_id']}"
            
            # Draw enhanced text information for each vehicle in the info panel
            draw.text((10, y_offset), 
                     f"{vehicle_label}: Speed={vehicle['velocity']:.2f} m/s, vx={vehicle['vx']:.2f}, vy={vehicle['vy']:.2f}, " +
                     f"Pos(x={vehicle['x_pos']:.2f}, y={vehicle['y_pos']:.2f})", 
                     fill=(255, 255, 255), font=font)
            y_offset += 15  # Increment y position for next text
        
        # Now draw the velocity arrows for the vehicles (without ID circles)
        for vehicle in sorted_vehicles:
            # Choose color based on vehicle type
            if vehicle['is_ego']:
                color = (255, 0, 0)  # Red for ego vehicle
            else:
                # Generate distinct colors for different vehicles
                colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
                         (255, 0, 255), (128, 128, 0), (0, 128, 128), (128, 0, 128)]
                color_idx = vehicle['display_id'] % len(colors)
                color = colors[color_idx]
            
            # Draw velocity arrow if the vehicle is moving
            if vehicle['velocity'] > 0.1:
                # Normalize velocity vector
                vx_norm = vehicle['vx'] / vehicle['velocity']
                vy_norm = vehicle['vy'] / vehicle['velocity']
                
                # Arrow end position (scaled by velocity)
                arrow_scale = min(30, vehicle['velocity'] * 7)  # Scale factor for arrow length
                end_x = int(vehicle['screen_x'] + vx_norm * arrow_scale)
                end_y = int(vehicle['screen_y'] - vy_norm * arrow_scale)  # Invert y-axis
                
                # Draw arrow
                draw.line((vehicle['screen_x'], vehicle['screen_y'], end_x, end_y), fill=color, width=2)
                
                # Draw arrowhead
                arrow_head_size = 7
                angle = np.arctan2(-vy_norm, vx_norm)  # Invert y for screen coordinates
                angle1 = angle + np.pi * 3/4
                angle2 = angle - np.pi * 3/4
                draw.line((end_x, end_y, 
                         int(end_x - arrow_head_size * np.cos(angle1)),
                         int(end_y - arrow_head_size * np.sin(angle1))), 
                         fill=color, width=2)
                draw.line((end_x, end_y, 
                         int(end_x - arrow_head_size * np.cos(angle2)),
                         int(end_y - arrow_head_size * np.sin(angle2))), 
                         fill=color, width=2)
        
        # Add metadata
        screen_height = 600  # Ensure this is defined
        draw.text((10, screen_height - 20), f"Episode: {episode_num}, Step: {step_num}", fill=(255, 255, 255), font=font)
        
        # Generate timestamp for unique filename
        timestamp = int(time.time() * 1000)
        
        # Save the image with timestamp for uniqueness
        filename = f"car_scene_ep{episode_num}_step{step_num}_{timestamp}.png"
        filepath = os.path.join(self.image_dir, filename)
        img.save(filepath)
        
        return filename
    
    def format_observation(self, observation: np.ndarray) -> str:
        """Format the observation into a readable string."""
        obs_str = "Current intersection environment state:\n"
        for i, vehicle in enumerate(observation):
            if i == 0:
                obs_str += "Your vehicle: "
            else:
                obs_str += f"Vehicle {i}: "
            
            # For intersection, the first column is 'presence'
            if len(vehicle) >= 7 and vehicle[0] > 0.5:  # Ensure vehicle is present
                obs_str += f"Presence={int(vehicle[0])}, "
                obs_str += f"Position (x={vehicle[1]:.2f}, y={vehicle[2]:.2f}), "
                obs_str += f"Velocity (vx={vehicle[3]:.2f}, vy={vehicle[4]:.2f}), "
                obs_str += f"Heading (cos_h={vehicle[5]:.2f}, sin_h={vehicle[6]:.2f})\n"
        
        return obs_str
    
    def define_action_mapping(self):
        """Define a mapping from action indexes to human-readable labels."""
        # For intersection environment with left turn + longitudinal control
        return {
            0: "SLOWER + LEFT",
            1: "IDLE + LEFT",
            2: "FASTER + LEFT",
            3: "SLOWER + IDLE",
            4: "IDLE + IDLE",
            5: "FASTER + IDLE",
            6: "SLOWER + RIGHT",
            7: "IDLE + RIGHT",
            8: "FASTER + RIGHT"
        }
    
    def suggest_best_move(self, obs: np.ndarray) -> str:
        """
        Suggest the best move based on the current observation.
        This is a more comprehensive heuristic considering various aspects of the scene.
        """
        action_map = self.define_action_mapping()
        
        # Extract ego vehicle data
        ego_x, ego_y = obs[0][1], obs[0][2]
        ego_vx, ego_vy = obs[0][3], obs[0][4]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
        
        # Default is turn left at moderate speed
        best_action = 1  # IDLE + LEFT
        
        # Initialize variables to track potential hazards
        closest_front_vehicle_dist = float('inf')
        vehicles_from_right = []
        vehicles_from_left = []
        vehicles_from_bottom = []
        
        # Analyze all other vehicles
        for i in range(1, len(obs)):
            if obs[i][0] > 0.5:  # If vehicle is present
                other_x, other_y = obs[i][1], obs[i][2]
                other_vx, other_vy = obs[i][3], obs[i][4]
                
                # Calculate relative position (vector from ego to other)
                rel_x = other_x - ego_x
                rel_y = other_y - ego_y
                
                # Simple distance calculation
                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                # Determine vehicle direction relative to ego
                # Simplified directional categorization
                if rel_y < -0.05 and abs(rel_x) < 0.2:  # Vehicle in front (below in y)
                    if distance < closest_front_vehicle_dist:
                        closest_front_vehicle_dist = distance
                elif rel_x > 0.05:  # Vehicle from right side
                    vehicles_from_right.append((distance, other_vx, other_vy))
                elif rel_x < -0.05:  # Vehicle from left side
                    vehicles_from_left.append((distance, other_vx, other_vy))
                elif rel_y > 0.05:  # Vehicle from bottom (approaching from behind)
                    vehicles_from_bottom.append((distance, other_vx, other_vy))
        
        # Decision making logic
        # 1. First priority: Check for vehicles directly in front
        if closest_front_vehicle_dist < 0.3:
            # Vehicle very close in front, slow down
            best_action = 0  # SLOWER + LEFT
        elif 0.3 <= closest_front_vehicle_dist < 0.5:
            # Vehicle moderately close in front, maintain speed
            best_action = 1  # IDLE + LEFT
        else:
            # No close vehicles in front
            # 2. Second priority: Check for cross traffic from right (higher danger)
            right_threat = any(dist < 0.4 and vx < 0 for dist, vx, vy in vehicles_from_right)
            
            # 3. Third priority: Check for traffic from left
            left_threat = any(dist < 0.4 and vx > 0 for dist, vx, vy in vehicles_from_left)
            
            if right_threat or left_threat:
                # Cross traffic nearby, slow down
                best_action = 0  # SLOWER + LEFT
            elif not vehicles_from_right and not vehicles_from_left and closest_front_vehicle_dist > 0.8:
                # Clear path ahead, speed up
                best_action = 2  # FASTER + LEFT
            else:
                # Default to maintain speed
                best_action = 1  # IDLE + LEFT
        
        # Get the text description of the selected action
        best_move = action_map[best_action]
        return best_move
    
    def capture_data(self, num_episodes: int = 5, num_steps_per_episode: int = 24, 
                   auto_label: bool = True, manual_label: bool = False):
        """
        Capture data from the environment.
        
        Args:
            num_episodes: Number of episodes to capture
            num_steps_per_episode: Maximum number of steps per episode
            auto_label: Whether to automatically label the best move
            manual_label: Whether to prompt for manual labeling
        """
        env = self.setup_environment()
        action_map = self.define_action_mapping()
        total_captures = 0
        
        # Get valid actions from the environment's action space
        valid_action_count = env.action_space.n
        print(f"Environment has {valid_action_count} valid actions")
        
        for episode in range(num_episodes):
            print(f"Starting episode {episode+1}/{num_episodes}")
            obs, info = env.reset()
            
            # For each episode, use a different default action to increase variety
            # Use only valid actions
            default_actions = [0, 1, 2]  # LANE_LEFT + (SLOWER, IDLE, FASTER)
            default_action_idx = episode % len(default_actions)
            default_action = default_actions[default_action_idx]
            
            # Wait a few steps to allow vehicles to populate the scene
            warmup_steps = 2
            for _ in range(warmup_steps):
                obs, reward, done, truncated, info = env.step(1)  # LANE_LEFT + IDLE
                if done or truncated:
                    break
            
            for step in range(num_steps_per_episode):
                # Use variety of actions to move the agent through the scene differently
                if step % 3 == 0:
                    # Every third step, try a different action
                    action = (default_action + step % valid_action_count) % valid_action_count
                else:
                    action = default_action
                
                # Try to capture multiple frames near the intersection
                if self.is_at_intersection_entrance(obs):
                    print(f"Agent at intersection entrance in episode {episode}, step {step}. Capturing frame...")
                    
                    # Capture frame before taking action
                    filename = self.capture_frame(env, obs, step, episode)
                    total_captures += 1
                    
                    # Generate observation text for labeling
                    obs_text = self.format_observation(obs)
                    
                    # Label the scene
                    if auto_label:
                        best_move = self.suggest_best_move(obs)
                        label = best_move
                    elif manual_label:
                        print(f"\nFrame: {filename}")
                        print(obs_text)
                        print("\nAvailable actions:")
                        for idx, action_name in action_map.items():
                            print(f"{idx}: {action_name}")
                        label = input("Enter the best move (action number or description): ")
                        if label.isdigit() and int(label) in action_map:
                            label = action_map[int(label)]
                    else:
                        label = "Unlabeled"
                    
                    # Store label
                    self.scene_labels.append({
                        "filename": filename,
                        "best_move": label,
                        "observation": obs.tolist(),  # Store the observation for future reference
                        "episode": episode,
                        "step": step
                    })
                
                # Take the action and step the environment
                obs, reward, done, truncated, info = env.step(action)
                
                if done or truncated:
                    print(f"Episode {episode+1} ended after {step+1} steps")
                    break
                
                # Small delay to not overwhelm the system
                time.sleep(0.1)
            
            print(f"Completed episode {episode+1}, total captures: {total_captures}")
        
        # Close the environment
        env.close()
        print(f"Data capture complete. Captured {total_captures} frames.")
        
        # Save the labels
        self.save_labels()
    
    def save_labels(self, filename: str = "scene_labels.json"):
        """Save the scene labels to a JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(self.scene_labels, f, indent=2)
            
        print(f"Saved {len(self.scene_labels)} scene labels to {filepath}")
    
    def load_labels(self, filename: str = "scene_labels.json"):
        """Load scene labels from a JSON file."""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.scene_labels = json.load(f)
            print(f"Loaded {len(self.scene_labels)} scene labels from {filepath}")
        else:
            print(f"Label file {filepath} not found.")
    
    def build_retrieval_index(self):
        """
        Build a simple retrieval index for the labeled scenes.
        This is a basic implementation that can be extended with vector embeddings.
        """
        if not self.scene_labels:
            print("No scene labels to index. Please capture data or load labels first.")
            return None
        
        # Simple dictionary-based index
        index = {
            "by_best_move": {},
            "by_episode": {},
            "by_step": {}
        }
        
        # Organize by best move
        for i, label in enumerate(self.scene_labels):
            best_move = label["best_move"]
            if best_move not in index["by_best_move"]:
                index["by_best_move"][best_move] = []
            index["by_best_move"][best_move].append(i)
            
            # Organize by episode
            episode = label["episode"]
            if episode not in index["by_episode"]:
                index["by_episode"][episode] = []
            index["by_episode"][episode].append(i)
            
            # Organize by step
            step = label["step"]
            if step not in index["by_step"]:
                index["by_step"][step] = []
            index["by_step"][step].append(i)
        
        return index
    
    def query_similar_scenes(self, observation: np.ndarray, top_k: int = 5):
        """
        Find similar scenes based on observation.
        This is a simple implementation using euclidean distance.
        
        Args:
            observation: The observation to compare against
            top_k: Number of top matches to return
            
        Returns:
            List of similar scene labels
        """
        if not self.scene_labels:
            print("No scene labels to query. Please capture data or load labels first.")
            return []
        
        # Flatten the observation for comparison
        flat_obs = observation.flatten()
        
        # Calculate distances to all stored observations
        distances = []
        for i, label in enumerate(self.scene_labels):
            stored_obs = np.array(label["observation"]).flatten()
            
            # Make sure the observations are the same length
            min_len = min(len(flat_obs), len(stored_obs))
            distance = np.linalg.norm(flat_obs[:min_len] - stored_obs[:min_len])
            distances.append((i, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Return top_k matches
        top_matches = [self.scene_labels[i] for i, _ in distances[:top_k]]
        return top_matches
    
    def manually_label_scenes(self):
        """
        Allow manual labeling or relabeling of captured scenes.
        
        This function displays each scene and prompts the user for a label.
        """
        if not self.scene_labels:
            print("No scene labels to edit. Please capture data or load labels first.")
            return
            
        action_map = self.define_action_mapping()
        
        print("\nAvailable actions:")
        for idx, action_name in action_map.items():
            print(f"{idx}: {action_name}")
        
        print("\nBeginning manual labeling process...")
        print("For each image, enter the number of the best action or 's' to skip.")
        
        for i, label in enumerate(self.scene_labels):
            filename = label["filename"]
            filepath = os.path.join(self.image_dir, filename)
            
            # Display observation details
            print(f"\n[{i+1}/{len(self.scene_labels)}] Scene: {filename}")
            print("Observation details:")
            obs_array = np.array(label["observation"])
            obs_text = self.format_observation(obs_array)
            print(obs_text)
            
            # Tell user to view the image file
            print(f"Please view the image at: {filepath}")
            print(f"Current label: {label['best_move']}")
            
            # Get user input
            user_input = input("Enter new label (action number or name), or 's' to skip: ")
            
            if user_input.lower() == 's':
                print("Skipping...")
                continue
            
            # Process the input
            if user_input.isdigit() and int(user_input) in action_map:
                new_label = action_map[int(user_input)]
                self.scene_labels[i]["best_move"] = new_label
                print(f"Updated label to: {new_label}")
            elif user_input in action_map.values():
                self.scene_labels[i]["best_move"] = user_input
                print(f"Updated label to: {user_input}")
            else:
                print("Invalid input. Keeping original label.")
        
        # Save updated labels
        self.save_labels()
        print("\nManual labeling complete. Labels saved.")

    def demonstrate_rag_integration(self, observation: np.ndarray):
        """
        Demonstrate how to integrate the labeled data with a RAG system.
        
        This function shows how to:
        1. Find similar scenes from the database
        2. Extract the best moves from those scenes
        3. Use these as examples or context for a model
        
        Args:
            observation: Current observation to find similar scenes for
            
        Returns:
            str: Suggested action based on similar past experiences
        """
        # Get similar scenes
        similar_scenes = self.query_similar_scenes(observation, top_k=3)
        
        if not similar_scenes:
            return "No similar scenes found. Using default action: IDLE + IDLE"
        
        # Extract best moves from similar scenes
        best_moves = [scene["best_move"] for scene in similar_scenes]
        
        # Count occurrences of each move
        move_counts = {}
        for move in best_moves:
            if move not in move_counts:
                move_counts[move] = 0
            move_counts[move] += 1
        
        # Find most common move
        most_common_move = max(move_counts.items(), key=lambda x: x[1])[0]
        
        # Format observation for LLM context
        obs_text = self.format_observation(observation)
        
        # Example of constructing a prompt for an LLM using RAG
        rag_prompt = f"""
Current Observation:
{obs_text}

Based on similar situations I've encountered before, the best moves were:
"""
        
        # Add similar scenes to the prompt
        for i, scene in enumerate(similar_scenes):
            rag_prompt += f"\nSimilar Scene {i+1}: {scene['filename']}\n"
            rag_prompt += f"Best Move: {scene['best_move']}\n"
        
        rag_prompt += f"\nMost common suggested move: {most_common_move}"
        
        # In a real system, you would send this prompt to an LLM
        print("\n=== RAG Integration Example ===")
        print(rag_prompt)
        print("===============================")
        
        return most_common_move

    def distribute_labeling_tasks(self, labelers=None):
        """
        Distribute labeling tasks among labelers.
        
        Args:
            labelers: List of labeler names. Defaults to ["Sean", "Simon", "Jack"]
        """
        if labelers is None:
            labelers = ["Sean", "Simon", "Jack"]
        
        if not self.scene_labels:
            print("No scene labels to distribute. Please capture data or load labels first.")
            return
        
        # Create directories for each labeler if they don't exist
        labeler_dirs = {}
        for labeler in labelers:
            labeler_dir = os.path.join(self.data_dir, f"labeler_{labeler}")
            labeler_image_dir = os.path.join(labeler_dir, "images")
            
            if not os.path.exists(labeler_dir):
                os.makedirs(labeler_dir)
            if not os.path.exists(labeler_image_dir):
                os.makedirs(labeler_image_dir)
            
            labeler_dirs[labeler] = {
                "dir": labeler_dir,
                "image_dir": labeler_image_dir
            }
        
        # Shuffle the scene labels for random distribution
        import random
        shuffled_labels = list(self.scene_labels)
        random.shuffle(shuffled_labels)
        
        # Calculate how many images each labeler should get
        images_per_labeler = len(shuffled_labels) // len(labelers)
        remainder = len(shuffled_labels) % len(labelers)
        
        print(f"Distributing {len(shuffled_labels)} images among {len(labelers)} labelers")
        print(f"Each labeler will get approximately {images_per_labeler} images")
        
        # Distribute the images
        labeler_assignments = {labeler: [] for labeler in labelers}
        
        current_index = 0
        for i, labeler in enumerate(labelers):
            # Calculate how many images this labeler should get
            count = images_per_labeler + (1 if i < remainder else 0)
            
            # Assign the images
            for j in range(count):
                if current_index < len(shuffled_labels):
                    image_data = shuffled_labels[current_index]
                    labeler_assignments[labeler].append(image_data)
                    
                    # Copy the image to the labeler's directory
                    src_image = os.path.join(self.image_dir, image_data["filename"])
                    dst_image = os.path.join(labeler_dirs[labeler]["image_dir"], image_data["filename"])
                    
                    # Use system copy to avoid potential issues with shutil
                    import shutil
                    shutil.copy2(src_image, dst_image)
                    
                    current_index += 1
        
        # Save the assignments to each labeler's directory
        for labeler, assignments in labeler_assignments.items():
            labeler_file = os.path.join(labeler_dirs[labeler]["dir"], "scene_labels.json")
            
            with open(labeler_file, "w") as f:
                json.dump(assignments, f, indent=2)
            
            print(f"Assigned {len(assignments)} images to {labeler}")
            print(f"Saved {labeler}'s assignments to {labeler_file}")
        
        return labeler_assignments

    def collect_and_merge_labels(self, labelers=None):
        """
        Collect and merge labels from multiple labelers.
        
        Args:
            labelers: List of labeler names. Defaults to ["Sean", "Simon", "Jack"]
        """
        if labelers is None:
            labelers = ["Sean", "Simon", "Jack"]
        
        all_labels = []
        
        for labeler in labelers:
            labeler_dir = os.path.join(self.data_dir, f"labeler_{labeler}")
            labeler_file = os.path.join(labeler_dir, "scene_labels.json")
            
            if os.path.exists(labeler_file):
                with open(labeler_file, "r") as f:
                    labels = json.load(f)
                
                print(f"Loaded {len(labels)} labels from {labeler}")
                
                # Add labeler information to each label
                for label in labels:
                    label["labeler"] = labeler
                
                all_labels.extend(labels)
            else:
                print(f"No labels found for {labeler} at {labeler_file}")
        
        # Save the merged labels
        if all_labels:
            merged_file = os.path.join(self.data_dir, "merged_labels.json")
            
            with open(merged_file, "w") as f:
                json.dump(all_labels, f, indent=2)
            
            print(f"Merged {len(all_labels)} labels from {len(labelers)} labelers")
            print(f"Saved merged labels to {merged_file}")
            
            # Update the scene labels
            self.scene_labels = all_labels
        
        return all_labels

def main():
    """Main function to capture and label data."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Capture and label intersection data.')
    parser.add_argument('--manual-label', action='store_true', 
                       help='Run manual labeling on existing captured data')
    parser.add_argument('--capture', action='store_true',
                       help='Capture new data even if existing data is found')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to capture')
    parser.add_argument('--distribute', action='store_true',
                       help='Distribute labeling tasks among labelers')
    parser.add_argument('--merge', action='store_true',
                       help='Merge labels from multiple labelers')
    parser.add_argument('--labelers', nargs='+', default=["Sean", "Simon", "Jack"],
                       help='Names of labelers to distribute tasks to')
    parser.add_argument('--target-count', type=int, default=120,
                       help='Target number of images to capture')
    parser.add_argument('--rag-query', action='store_true',
                       help='Demonstrate RAG query with a sample observation')
    args = parser.parse_args()
    
    data_capture = IntersectionDataCapture()
    
    # Check if manual labeling is requested
    if args.manual_label:
        # Load existing data if available
        if os.path.exists(os.path.join(data_capture.data_dir, "scene_labels.json")):
            print("Loading existing scene labels for manual labeling...")
            data_capture.load_labels()
            data_capture.manually_label_scenes()
        else:
            print("No existing data found. Please capture data first.")
            args.capture = True  # Force capture if no data exists
    
    # Check if we should capture data
    if args.capture or not os.path.exists(os.path.join(data_capture.data_dir, "scene_labels.json")):
        if args.capture:
            print("Capturing new data as requested...")
        else:
            print("No existing data found. Capturing new data...")
        
        # Calculate how many episodes to run to get approximately the target count of images
        # On average, each episode yields about 5 images at intersection entrances
        avg_images_per_episode = 5
        episodes_needed = max(5, args.target_count // avg_images_per_episode)
        
        print(f"Targeting {args.target_count} images, running approximately {episodes_needed} episodes")
        
        # We'll continue capturing until we reach the target count or max episodes
        max_episodes = episodes_needed * 2  # Set a maximum to prevent infinite loops
        total_captures = 0
        episodes_run = 0
        
        while total_captures < args.target_count and episodes_run < max_episodes:
            # Capture in smaller batches for more frequent updates
            batch_size = min(5, max_episodes - episodes_run)
            print(f"Running {batch_size} episodes (total {episodes_run}/{max_episodes})...")
            
            # Get the current count of labels
            pre_count = len(data_capture.scene_labels)
            
            # Capture data for a batch of episodes
            data_capture.capture_data(
                num_episodes=batch_size, 
                num_steps_per_episode=24, 
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
    
    elif not args.manual_label:  # Only load if not already loaded for manual labeling
        print("Loading existing scene labels...")
        data_capture.load_labels()
    
    # Distribute labeling tasks if requested
    if args.distribute:
        print(f"Distributing labeling tasks among {args.labelers}...")
        data_capture.distribute_labeling_tasks(args.labelers)
    
    # Merge labels if requested
    if args.merge:
        print(f"Merging labels from {args.labelers}...")
        data_capture.collect_and_merge_labels(args.labelers)
    
    # Demonstrate RAG query if requested
    if args.rag_query:
        # Build retrieval index if not already built
        index = data_capture.build_retrieval_index()
        
        # Create a new environment to get a fresh observation
        env = data_capture.setup_environment()
        new_obs, _ = env.reset()
        
        # Use the RAG integration to suggest an action
        print("\nDemonstrating RAG integration with a sample observation...")
        suggested_action = data_capture.demonstrate_rag_integration(new_obs)
        print(f"\nFinal suggested action based on RAG: {suggested_action}")
        
        # Clean up
        env.close()
    elif not any([args.manual_label, args.distribute, args.merge, args.capture]):
        # If no specific actions were requested, build the index anyway
        data_capture.build_retrieval_index()
    
    print("Data capture and labeling complete.")

if __name__ == "__main__":
    main() 