import os
import sys
import gymnasium as gym
import highway_env
import time
import argparse
import json
import numpy as np
import math
import pygame  # Add pygame import for overlay

# File to store the leaderboard
LEADERBOARD_FILE = "highway_leaderboard.json"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Manual control for Highway Environment')
    parser.add_argument('--action-type', type=str, choices=['discrete', 'continuous'], default='discrete',
                        help='Action type: discrete (lane changes & speed) or continuous (steering & acceleration)')
    parser.add_argument('--vehicles', type=int, default=30, help='Number of vehicles in the environment')
    parser.add_argument('--duration', type=int, default=40, help='Episode duration in seconds')
    parser.add_argument('--env', type=str, default='highway-v0', 
                      choices=['highway-v0', 'merge-v0', 'intersection-v0', 'roundabout-v0'],
                      help='Environment to use')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    parser.add_argument('--window-size', type=int, nargs=2, default=[800, 600], help='Window size (width, height)')
    parser.add_argument('--zoom', type=float, default=5.5, help='Zoom level (higher values zoom in more)')
    parser.add_argument('--clear_leaderboard', action='store_true', help='Clear the leaderboard')
    return parser.parse_args()

def get_vehicle_state_text(obs):
    """Generate text about vehicle state and nearby vehicles"""
    if obs is None or len(obs) == 0:
        return ["No observation data"]
        
    ego_vehicle = obs[0]  # First vehicle is always the ego vehicle
    
    # Extract ego vehicle info
    x, y = ego_vehicle[0], ego_vehicle[1]
    vx, vy = ego_vehicle[2], ego_vehicle[3]
    speed = np.sqrt(vx**2 + vy**2)
    
    # Determine lane position
    lane_text = "Unknown"
    if -0.2 < y < 0.2:
        lane_text = "Center Lane"
    elif y >= 0.2:
        lane_text = "Left Lane"
    elif y <= -0.2:
        lane_text = "Right Lane"
    
    # Get nearby vehicles
    nearby_vehicles = []
    for i in range(1, len(obs)):
        vehicle = obs[i]
        # Skip if all zeros (padding)
        if np.all(vehicle == 0):
            continue
            
        veh_x, veh_y = vehicle[0], vehicle[1]
        veh_vx, veh_vy = vehicle[2], vehicle[3]
        rel_speed = np.sqrt(veh_vx**2 + veh_vy**2) - speed
        
        # Determine position
        position = ""
        if abs(veh_y) < 0.2:  # Same lane
            position = "ahead" if veh_x > 0 else "behind"
        elif veh_y > 0.2:  # Left lane
            position = "left"
            if veh_x > 0:
                position += "-ahead"
            elif veh_x < 0:
                position += "-behind"
        elif veh_y < -0.2:  # Right lane
            position = "right"
            if veh_x > 0:
                position += "-ahead"
            elif veh_x < 0:
                position += "-behind"
                
        # Distance
        distance = np.sqrt(veh_x**2 + veh_y**2)
        
        # Add to list if close enough
        if distance < 2.0:  # Only show nearby vehicles
            nearby_vehicles.append(f"Vehicle {i}: {position}, distance: {distance:.1f}, rel. speed: {rel_speed:.1f}")
    
    # Create display text
    text_lines = [
        f"Speed: {speed:.2f}",
        f"Lane: {lane_text}",
        f"Position: ({x:.2f}, {y:.2f})",
        "Nearby Vehicles:"
    ]
    text_lines.extend(nearby_vehicles)
    
    return text_lines

def print_debug_info(obs):
    """Print debug information to console"""
    text_lines = get_vehicle_state_text(obs)
    print("\n=== DEBUG INFO ===")
    for line in text_lines:
        print(line)
    print("=================")

def load_leaderboard():
    """Load the leaderboard from file or create a new one"""
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Default leaderboard if file doesn't exist or is invalid
    return [
        {"name": "CPU", "distance": 0.0},
        {"name": "CPU", "distance": 0.0},
        {"name": "CPU", "distance": 0.0},
        {"name": "CPU", "distance": 0.0},
        {"name": "CPU", "distance": 0.0}
    ]

def save_leaderboard(leaderboard):
    """Save the leaderboard to file"""
    # Convert any NumPy types to native Python types before JSON serialization
    converted_leaderboard = []
    for entry in leaderboard:
        converted_entry = {
            "name": entry["name"],
            "distance": float(entry["distance"])  # Convert NumPy float32 to Python float
        }
        converted_leaderboard.append(converted_entry)
    
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(converted_leaderboard, f)

def update_leaderboard(leaderboard, distance):
    """Check if the current distance is in the top 5 and return its position (0-4), or -1 if not"""
    # If distance is 0, don't add to leaderboard
    if distance <= 0:
        return -1
        
    # Make sure the leaderboard is sorted by distance in descending order
    sorted_leaderboard = sorted(leaderboard, key=lambda entry: entry["distance"], reverse=True)
    
    # If the leaderboard has default entries with 0.0 distance, any positive distance should qualify
    has_default_entries = any(entry["name"] == "CPU" and entry["distance"] == 0.0 for entry in sorted_leaderboard)
    
    if has_default_entries:
        # Find the first default entry (CPU with 0.0)
        position = 0
        for i, entry in enumerate(sorted_leaderboard):
            if entry["name"] == "CPU" and entry["distance"] == 0.0:
                position = i
                break
        return position
    
    # Regular case: check if the score is better than any existing score
    if len(sorted_leaderboard) >= 5 and distance < sorted_leaderboard[4]["distance"]:
        return -1  # Not in top 5
    
    # Find position to insert based on distance (descending order)
    position = 0
    while position < len(sorted_leaderboard) and distance < sorted_leaderboard[position]["distance"]:
        position += 1
    
    return position if position < 5 else -1

def display_leaderboard(leaderboard):
    """Display the leaderboard in a nice format"""
    # Sort the leaderboard by distance in descending order
    sorted_leaderboard = sorted(leaderboard, key=lambda entry: entry["distance"], reverse=True)
    
    print("\n=== TOP DISTANCES ===")
    print("Rank | Name      | Distance")
    print("-" * 30)
    
    for i, entry in enumerate(sorted_leaderboard[:5]):
        print(f" {i+1}   | {entry['name']:<10} | {entry['distance']:.3f}")
    
    print("-" * 30)

def is_out_of_bounds(obs):
    """Check if the ego vehicle is out of bounds (outside highway lanes)"""
    if obs is None or len(obs) == 0:
        return False
    
    ego_vehicle = obs[0]
    x, y = ego_vehicle[0], ego_vehicle[1]  # Get both x and y positions
    
    # Define stricter boundaries for the highway
    # Upper boundary (left side of highway)
    # Based on visual inspection, the top white line is around y=0.8
    upper_boundary = 0.85
    
    # Lower boundary (right side of highway)
    # Based on visual inspection, the bottom white line is around y=-0.8
    lower_boundary = -0.12
    
    # Check if the vehicle is outside the boundaries
    if y > upper_boundary or y < lower_boundary:
        print(f"DEBUG: Vehicle out of bounds at y={y:.2f} (limits: {lower_boundary:.2f} to {upper_boundary:.2f})")
        return True
    
    return False

def has_crashed(obs, info):
    """Check if the ego vehicle has crashed"""
    # First check the info dict for explicit crash indication
    if info.get('crashed', False):
        # Even if environment reports crash, verify it's not just heavy braking
        # This is a workaround for highway-env sometimes incorrectly reporting crashes
        if obs is not None and len(obs) > 0:
            ego_vx, ego_vy = obs[0][2], obs[0][3]
            ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
            # If still moving with reasonable speed, likely not a crash
            if ego_speed > 0.1:
                # Check if we have a nearby vehicle before confirming crash
                has_nearby_vehicle = False
                for i in range(1, len(obs)):
                    if np.all(obs[i] == 0):  # Skip empty entries
                        continue
                    veh_x, veh_y = obs[i][0], obs[i][1]
                    dist = np.sqrt(veh_x**2 + veh_y**2)
                    if dist < 0.15:  # Vehicle is very close
                        has_nearby_vehicle = True
                        break
                # Only treat as crash if there's a nearby vehicle
                return has_nearby_vehicle
        return True
    
    # Check for collisions in the observation (overlapping vehicles)
    if obs is None or len(obs) < 2:  # Need at least ego + one other vehicle
        return False
    
    ego_vehicle = obs[0]
    ego_x, ego_y = ego_vehicle[0], ego_vehicle[1]
    ego_vx, ego_vy = ego_vehicle[2], ego_vehicle[3]
    ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
    
    # Only check for crashes if we can get valid ego vehicle data
    if np.all(ego_vehicle == 0):
        return False
    
    # If we're braking heavily, raise collision threshold to avoid false positives
    is_braking = False
    if hasattr(ego_vehicle, 'action') and len(ego_vehicle.action) >= 2:
        is_braking = ego_vehicle.action[1] < 0.4  # Consider braking if acceleration is low
    
    # Alternative braking detection - check for rapid deceleration
    prev_speed = getattr(has_crashed, 'prev_speed', None)
    if prev_speed is not None:
        deceleration = prev_speed - ego_speed
        is_braking = is_braking or (deceleration > 0.05)  # Consider significant deceleration as braking
    
    # Store current speed for next frame
    has_crashed.prev_speed = ego_speed
    
    # Determine base collision threshold
    base_collision_threshold = 0.05  # Normal threshold
    
    for i in range(1, len(obs)):
        vehicle = obs[i]
        if np.all(vehicle == 0):  # Skip empty entries
            continue
        
        veh_x, veh_y = vehicle[0], vehicle[1]
        veh_vx, veh_vy = vehicle[2], vehicle[3]
        
        # Calculate distance between vehicles
        dist = np.sqrt((ego_x - veh_x)**2 + (ego_y - veh_y)**2)
        
        # Calculate relative velocity to determine if this is a real crash
        rel_vx = ego_vx - veh_vx
        rel_vy = ego_vy - veh_vy
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
        
        # If braking, use higher collision threshold and additional checks
        collision_threshold = base_collision_threshold
        if is_braking:
            # Significantly increase threshold during braking to avoid false crashes
            collision_threshold = 0.03  # Stricter threshold during braking
            
            # During braking, require higher relative speed for crash detection
            min_rel_speed_for_crash = 0.5  # Higher threshold during braking
        else:
            # During normal driving, use regular threshold
            min_rel_speed_for_crash = 0.2
        
        # Only register as crash if:
        # 1. Vehicles are very close (below threshold)
        # 2. Relative speed is significant (to avoid false positives during parallel driving or braking)
        if dist < collision_threshold and rel_speed > min_rel_speed_for_crash:
            # Additional check: confirm crash only if vehicles are moving towards each other
            # Calculate dot product of relative position and relative velocity
            rel_pos_x, rel_pos_y = veh_x - ego_x, veh_y - ego_y
            moving_towards = rel_pos_x * rel_vx + rel_pos_y * rel_vy
            
            # If vehicles are moving towards each other, it's likely a crash
            if moving_towards < 0:
                return True
    
    return False

def get_input_with_timeout(prompt, timeout=None):
    """Get input from user with or without a timeout"""
    
    # If timeout is None, use regular input for infinite time
    if timeout is None:
        return input(f"{prompt}: ") or "Anonymous"
    
    # For timed input, print the prompt first
    print(prompt)
    
    # Store original terminal settings
    import termios
    import fcntl
    import os
    import select
    import sys
    
    fd = sys.stdin.fileno()
    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)
    
    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
    
    try:
        name = ""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to read from stdin
                c = sys.stdin.read(1)
                if c == '\n':
                    break
                elif c == '\x7f':  # Backspace
                    if name:
                        name = name[:-1]
                        # Erase character from terminal
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()
                elif c:
                    name += c
                    # Echo character
                    sys.stdout.write(c)
                    sys.stdout.flush()
            except IOError:
                pass
            
            # Display remaining time
            remaining = int(timeout - (time.time() - start_time))
            sys.stdout.write(f"\r{prompt} ({remaining}s remaining): {name}")
            sys.stdout.flush()
            time.sleep(0.1)
        
        print()  # New line after input
        return name if name else "Anonymous"
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

def render_distance_overlay(env, distance):
    """Render distance overlay on the screen using a simpler approach"""
    try:
        # Get pygame display surface
        screen = pygame.display.get_surface()
        
        if screen is None:
            # Try to initialize pygame if not already done
            if not pygame.get_init():
                pygame.init()
                screen = pygame.display.get_surface()
                if screen is None:
                    print("Warning: No pygame display available for overlay")
                    return
        
        # Create font object
        font = pygame.font.Font(None, 36)  # Default font, size 36
        
        # Create text surface with distance
        text = font.render(f"Distance: {distance:.3f}", True, (255, 255, 255))
        
        # Position the text in the top-right corner
        text_rect = text.get_rect(topright=(screen.get_width() - 10, 10))
        
        # Add semi-transparent background
        s = pygame.Surface((text_rect.width + 10, text_rect.height + 10), pygame.SRCALPHA)
        s.fill((0, 0, 0, 128))
        screen.blit(s, (text_rect.x - 5, text_rect.y - 5))
        
        # Draw text on screen
        screen.blit(text, text_rect)
        
        # Force an update of the screen for this region to make sure it's visible
        pygame.display.update(pygame.Rect(text_rect.x - 5, text_rect.y - 5, 
                                        text_rect.width + 10, text_rect.height + 10))
    except Exception as e:
        # In case of any rendering errors, just log and continue
        print(f"Warning: Could not render overlay: {str(e)}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Clear leaderboard if requested
    if args.clear_leaderboard:
        clear_leaderboard()
    
    # Initialize pygame for rendering overlay
    pygame.init()
    
    # Configure environment based on action type
    config = {
        # Enable manual control from the start
        "manual_control": True,
        # Real-time rendering is required for manual control
        "real_time_rendering": True,
        
        # Set screen dimensions
        "screen_width": args.window_size[0],
        "screen_height": args.window_size[1],
        
        # Set zoom/scaling factor
        "scaling": args.zoom,
        
        # Vehicle settings
        "vehicles_count": args.vehicles,
        "vehicles_density": 1.5,  # Increased density for more challenging gameplay
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        
        # Make other vehicles more aggressive
        "initial_spacing": 1.5,  # Reduced spacing between vehicles
        "collision_reward": -1,  # Increase penalty for collisions
        
        # Additional vehicle behavior parameters
        "other_vehicles_distribution": [
            {
                # Standard cars
                "proportion": 0.7,
                "vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "kwargs": {
                    "politeness": 0.0,  # Reduced politeness (more aggressive)
                    "lane_change_min_acc_gain": 0.05,  # More willing to change lanes
                    "allow_lane_changes": True
                }
            },
            {
                # Fast and aggressive cars
                "proportion": 0.2,
                "vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "kwargs": {
                    "v0": 35,  # Higher target speed
                    "politeness": -0.5,  # Negative politeness (very aggressive)
                    "lane_change_min_acc_gain": 0.0,  # Will change lanes even if no gain
                    "allow_lane_changes": True
                }
            },
            {
                # Erratic drivers
                "proportion": 0.1,
                "vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "kwargs": {
                    "v0": 30,  # Variable speed
                    "politeness": -1.0,  # Very impolite
                    "lane_change_min_acc_gain": -0.5,  # Will change lanes randomly
                    "allow_lane_changes": True
                }
            }
        ],
        
        # Environment settings
        "duration": args.duration,
        
        # Observation settings
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,  # Increased observation count to see more vehicles
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        }
    }
    
    # Set action type based on command line argument
    if args.action_type == 'continuous':
        config["action"] = {
            "type": "ContinuousAction"
        }
        control_instructions = [
            "CONTINUOUS CONTROL MODE:",
            "Arrow Keys: Control steering and acceleration",
            "← → (Left/Right Arrows): Steering",
            "↑ ↓ (Up/Down Arrows): Acceleration/Braking"
        ]
    else:  # discrete
        config["action"] = {
            "type": "DiscreteMetaAction"
        }
        control_instructions = [
            "DISCRETE CONTROL MODE:",
            "← (Left Arrow): Change to left lane",
            "→ (Right Arrow): Change to right lane",
            "↑ (Up Arrow): Increase speed",
            "↓ (Down Arrow): Decrease speed"
        ]
    
    # Create the environment with manual control enabled from the start
    env = gym.make(
        args.env,
        render_mode="human",
        config=config
    )
    
    # Create a wrapper for the environment's render method to add the overlay
    original_render = env.render
    total_distance_for_render = [0.0]  # Use a list to make it mutable from the closure
    
    def render_with_overlay():
        # Call the original render method
        result = original_render()
        
        # Then render our overlay
        try:
            screen = pygame.display.get_surface()
            if screen is not None:
                # Create font object
                font = pygame.font.Font(None, 36)
                
                # Create text surface
                text = font.render(f"Distance: {total_distance_for_render[0]:.3f}", True, (255, 255, 255))
                
                # Position in top-right corner
                text_rect = text.get_rect(topright=(screen.get_width() - 10, 10))
                
                # Semi-transparent background
                s = pygame.Surface((text_rect.width + 10, text_rect.height + 10), pygame.SRCALPHA)
                s.fill((0, 0, 0, 128))
                screen.blit(s, (text_rect.x - 5, text_rect.y - 5))
                
                # Draw text
                screen.blit(text, text_rect)
                
                # Update this region
                pygame.display.update(pygame.Rect(text_rect.x - 5, text_rect.y - 5, 
                                                text_rect.width + 10, text_rect.height + 10))
        except Exception as e:
            print(f"Warning: Could not render overlay: {str(e)}")
            
        return result
    
    # Replace the render method
    env.render = render_with_overlay
    
    # Reset the environment to initialize
    obs, info = env.reset()
    
    print(f"\n=== Highway Manual Control - {args.env} - HARD MODE ===")
    print(f"Action Type: {args.action_type.upper()}")
    print(f"Traffic Density: HIGH ({args.vehicles} vehicles with aggressive driving)")
    for instruction in control_instructions:
        print(instruction)
    print("Press ESC to quit")
    print("Press 'd' to toggle debug information (in console)")
    print("===============================\n")
    
    # Initialize scoring
    total_distance = 0.0
    prev_position = [0, 0]  # Start position
    if len(obs) > 0:
        prev_position = [obs[0][0], obs[0][1]]
    
    # Main game loop
    done = truncated = False
    crashed = False
    show_debug = args.debug
    debug_counter = 0  # Counter to avoid printing debug too frequently
    speed = 0.0
    forward_progress = 0.0
    
    # Force maximum speed when starting
    if args.action_type == 'continuous':
        # For continuous control, apply max acceleration at the start
        # We'll allow some braking but prevent stopping
        print("High speed mode active - moderate braking allowed, but no stopping")
    else:
        # For discrete control, we'll also maintain high speed but allow some slowing
        print("High speed mode active - moderate braking allowed, but no stopping")
    
    # Note: We've removed the event handler override since there's no window attribute
    # Instead, we'll rely on overriding the actions in the step function
    
    while not (done or truncated or crashed):
        # For continuous actions, allow some braking but maintain minimum speed
        if args.action_type == 'continuous':
            # Start with default action
            forced_action = np.array([0.0, 1.0])  # [steering=0, acceleration=1.0]
            
            # Get current action from environment (for both steering and acceleration)
            if hasattr(env, 'action'):
                current_action = env.action
                if isinstance(current_action, np.ndarray) and len(current_action) >= 2:
                    # Keep steering input
                    forced_action[0] = current_action[0]
                    
                    # Allow some braking but maintain a minimum acceleration
                    # Limit acceleration to range [0.3, 1.0]
                    # This allows braking but prevents stopping completely
                    accel = current_action[1]
                    forced_action[1] = max(0.3, accel)  # Minimum 0.3 acceleration
            
            # Apply the modified action
            env.action = forced_action
            obs, reward, done, truncated, info = env.step(forced_action)
        else:
            # For discrete actions, allow SLOWER action (1) but not IDLE (4)
            # Get original action from environment if available
            action = 2  # Default: FASTER
            
            if hasattr(env, 'action'):
                current_action = env.action
                
                # Allow SLOWER (1) but don't allow IDLE (4) or actions that would stop the car
                if current_action == 1:  # SLOWER
                    action = 1
                elif current_action == 3:  # LANE_LEFT
                    action = 3
                elif current_action == 5:  # LANE_RIGHT
                    action = 5
                else:
                    action = 2  # Default to FASTER for other actions
            
            # Apply the action
            env.action = action
            obs, reward, done, truncated, info = env.step(action)
        
        # IMMEDIATELY check if vehicle is out of bounds
        if obs is not None and len(obs) > 0 and is_out_of_bounds(obs):
            print("\n*** OUT OF BOUNDS! Vehicle crashed by leaving the highway. ***")
            crashed = True
            done = True
            break
        
        # Calculate distance traveled
        if len(obs) > 0:
            current_position = [obs[0][0], obs[0][1]]
            
            # Get velocity from observation
            vx, vy = obs[0][2], obs[0][3]
            speed = math.sqrt(vx**2 + vy**2)
            
            # Calculate forward distance (mainly x-direction movement)
            # Scale by simulation time step - use a larger factor to make progress more visible
            # Highway-env typically uses small scale for coordinates
            simulation_step = 0.1  # Adjust this based on environment step size
            forward_progress = max(0, speed * simulation_step)
            
            # Add to total distance - only when actually moving forward
            if speed > 0.1:  # Small threshold to avoid counting tiny movements
                total_distance += forward_progress
                # Update the reference for the render method
                total_distance_for_render[0] = total_distance
            
            prev_position = current_position
        
        # Check terminal conditions separately to provide better messages
        
        # Terminal conditions from environment
        if done:
            if info.get('crashed', False):
                # Double-check with our own crash detection for verification
                if has_crashed(obs, info):
                    crashed = True
                    print("\n*** CRASH DETECTED! ***")
                else:
                    # This might be a false positive during braking
                    print("\n*** SIMULATION ENDED - Environment detected crash but not confirmed ***")
            else:
                print("\n*** SIMULATION ENDED - Goal reached or time expired ***")
        
        if truncated:
            print("\n*** SIMULATION TRUNCATED ***")
        
        # Our own crash detection - with improved logic for braking situations
        if not crashed and has_crashed(obs, info):
            crashed = True
            print("\n*** CRASH DETECTED! ***")
        
        # Print debug info if enabled (but not every frame to avoid console spam)
        if show_debug and debug_counter % 30 == 0:  # Every ~1 second at 30 FPS
            print_debug_info(obs)
            print(f"Current distance: {total_distance:.3f}")
            print(f"Current speed: {speed:.3f}, Forward progress: {forward_progress:.3f}")
            
            # Add debug info about environment state
            if hasattr(env, 'action'):
                print(f"Current action: {env.action}")
            if hasattr(info, 'crashed'):
                print(f"Environment crash state: {info.get('crashed', False)}")
            
            # Add boundary debug info
            if len(obs) > 0:
                y = obs[0][1]
                lane_width = 0.5
                num_lanes = 3
                min_y = -lane_width * (num_lanes / 2)
                max_y = lane_width * (num_lanes / 2)
                right_leniency = 0.05
                left_leniency = 0.05
                print(f"Lane position: y={y:.3f}, Boundaries: left={max_y+left_leniency:.3f}, right={min_y-right_leniency:.3f}")
        
        debug_counter += 1
        
        # Check for 'd' key press for debug toggle (using 'info' from environment)
        if 'd' in info.get('key_pressed', '') and debug_counter % 15 == 0:
            show_debug = not show_debug
            if show_debug:
                print("\nDebug display ENABLED")
            else:
                print("\nDebug display DISABLED")
                
        # Brief pause to avoid hogging CPU
        time.sleep(0.01)
    
    # Clean up
    env.close()
    
    # Display final score
    print(f"\n=== GAME OVER ===")
    print(f"Distance traveled: {total_distance:.3f}")
    
    # Check if score is in top 5
    leaderboard = load_leaderboard()
    
    # Sort the leaderboard first to ensure proper ranking
    leaderboard = sorted(leaderboard, key=lambda entry: entry["distance"], reverse=True)
    
    position = update_leaderboard(leaderboard, total_distance)
    
    # Display the leaderboard
    display_leaderboard(leaderboard)
    
    # If score is in top 5, get user name and update leaderboard
    if position >= 0:
        print(f"\nCongratulations! You made it onto the leaderboard!")
        name = get_input_with_timeout("Enter your name", None)  # None for infinite time
        
        # Insert the new score at the right position
        # Convert distance to native Python float to avoid NumPy type issues
        new_entry = {"name": name, "distance": float(total_distance)}
        
        # Add the new entry
        leaderboard.append(new_entry)
        
        # Re-sort to ensure the correct order
        leaderboard = sorted(leaderboard, key=lambda entry: entry["distance"], reverse=True)
        
        # Keep only top 5
        if len(leaderboard) > 5:
            leaderboard = leaderboard[:5]
        
        # Save the updated leaderboard
        save_leaderboard(leaderboard)
        
        # Display the updated leaderboard
        print("\nUpdated leaderboard:")
        display_leaderboard(leaderboard)

    # No need for final overlay render - environment is already closed
    # render_distance_overlay(env, total_distance)

if __name__ == "__main__":
    main() 