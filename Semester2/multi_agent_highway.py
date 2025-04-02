import os
import gymnasium
import highway_env
from matplotlib import pyplot as plt
from openai import OpenAI
import numpy as np
import time
import json
from datetime import datetime
#from rag import HighwayRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY_SIMON")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Create log directory if it doesn't exist
log_dir = "driving_logs"
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/driving_session_{timestamp}.log"

# Log the start of the session
with open(log_filename, "w") as f:
    f.write(f"=== Driving Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

def log_message(message, agent_id=None):
    """Log a message to the log file with a timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    prefix = f"[{timestamp}]"
    if agent_id is not None:
        prefix += f" [Agent {agent_id}]"
    
    with open(log_filename, "a") as f:
        f.write(f"{prefix} {message}\n")

# Initialize RAG system
#rag = HighwayRAG(openai_api_key=api_key)

# Create the highway environment with multiple agents
env = gymnasium.make(
    "highway-v0",
    render_mode="human",  # Changed from rgb_array to human
    config={
        # Control total number of vehicles in the environment
        "vehicles_count": 10,  # Reduced number of vehicles to reduce complexity
        "vehicles_density": 1,  # Reduced density to spread vehicles out more
        
        # Multi-agent configuration
        "controlled_vehicles": 5,  # Reduced number of controlled vehicles
        
        # Set the initial speed for all vehicles
        "initial_lane_id": None,  # Let the environment handle lane assignment
        "duration": 40,  # Episode duration in seconds
        
        # Show controlled vehicles more clearly
        "show_trajectories": True,  # Show trajectories of controlled vehicles
        
        # Vehicle colors for better visibility
        "vehicles_colors": {
            0: "#FF0000",  # Red for first controlled vehicle
            1: "#00FF00",  # Green for second controlled vehicle
            2: "#0000FF",  # Blue for third controlled vehicle
        },
        
        # Make controlled vehicles more visible
        "controlled_vehicles_highlight": True,  # Highlight controlled vehicles
        
        # Vehicle sizes for better visibility
        "vehicles_sizes": {
            0: 1.5,  # Make first controlled vehicle larger
            1: 1.5,  # Make second controlled vehicle larger
            2: 1.5,  # Make third controlled vehicle larger
        },
        
        # Camera configuration
        "camera": {
            "type": "BirdEyeCamera",
            "position": [0, 0, 200],  # Much higher position to see more of the road
            "rotation": [-90, 0, 0],  # Top-down view
            "fov": 160,  # Very wide field of view to see more of the road
        },
        
        # Multi-agent observation space
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 10,  # Increased from 5 to 10 observable vehicles
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
            }
        },
        
        # Multi-agent action space
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
            }
        },
        
        # Screen dimensions and zoom
        "screen_width": 1200,  # Wider screen
        "screen_height": 800,  # Taller screen
        "scaling": 5.0,  # Zoom out to see more of the road
    }
)

log_message("Environment created with configuration:")
log_message(f"Number of controlled vehicles: 3")
log_message(f"Other vehicles: 7")
log_message(f"Camera type: BirdEyeCamera")

# Clear explanation log files at the start of each run
num_agents = env.unwrapped.config["controlled_vehicles"]
log_message(f"Clearing explanation logs for {num_agents} agents")
for i in range(num_agents):
    # Create or truncate (clear) the explanation log file
    with open(f"explanations_log_agent_{i}.txt", "w") as f:
        f.write(f"=== Explanation Log for Agent {i} - Session: {timestamp} ===\n\n")
    log_message(f"Cleared explanation log file", agent_id=i)

# Reset the environment and get initial observations
obs, info = env.reset()

# Print the observation structure to verify multi-agent setup
print("Observation structure:", type(obs))
print("Number of observations:", len(obs))
for i, obs_i in enumerate(obs):
    print(f"Observation for vehicle {i}:", obs_i.shape)
    log_message(f"Initial observation shape: {obs_i.shape}", agent_id=i)

def parse_observation(observation, agent_idx):
    """
    Parse and describe the observation in human-readable format
    """
    # Get ego vehicle info (first row is always the ego vehicle)
    ego_vehicle = observation[0]
    
    # Extract information about nearby vehicles
    nearby_vehicles = []
    for i in range(1, len(observation)):
        vehicle = observation[i]
        x, y = vehicle[0], vehicle[1]
        vx, vy = vehicle[2], vehicle[3]
        
        # Skip vehicles with all zeros (padding)
        if np.all(vehicle == 0):
            continue
            
        # Determine the relative position
        if abs(y) < 0.2:
            lane = "same lane"
        elif y > 0:
            lane = "left lane"
        else:
            lane = "right lane"
            
        # Determine relative position
        if x > 0:
            position = "ahead"
        else:
            position = "behind"
            
        # Determine relative speed
        if vx > 0:
            speed = "faster"
        else:
            speed = "slower"
            
        distance = abs(x)
        
        nearby_vehicles.append({
            "position": position,
            "lane": lane,
            "distance": f"{distance:.2f}",
            "relative_speed": speed,
            "coordinates": f"x={x:.2f}, y={y:.2f}",
            "velocity": f"vx={vx:.2f}, vy={vy:.2f}"
        })
    
    return {
        "agent_id": agent_idx,
        "nearby_vehicles": nearby_vehicles,
        "num_nearby_vehicles": len(nearby_vehicles)
    }

def predict_action(observations):
    """Predict actions for all agents based on their observations"""
    # All available actions in highway environment
    available_actions = {
        0: 'LANE_LEFT (change to left lane)',
        1: 'IDLE (maintain current lane position)',
        2: 'LANE_RIGHT (change to right lane)',
        3: 'FASTER (accelerate)',
        4: 'SLOWER (decelerate)'
    }
    
    # Create a detailed explanation of the observation format to help the LLM understand
    observation_explanation = (
        "The observation is a 2D array where each row represents a vehicle and columns represent features:\n"
        "- Column 0: x (position along road axis, normalized)\n"
        "- Column 1: y (position across road axis, normalized)\n"
        "- Column 2: vx (velocity along road axis, normalized)\n"
        "- Column 3: vy (velocity across road axis, normalized)\n"
        "- Column 4: cos_h (cosine of heading angle)\n"
        "- Column 5: sin_h (sine of heading angle)\n\n"
        "Row 0 is YOUR vehicle. Remaining rows are other vehicles, sorted by distance (closest first).\n"
        "Coordinates are relative to your vehicle, so:\n"
        "- x > 0: vehicle is ahead of you\n"
        "- x < 0: vehicle is behind you\n"
        "- y > 0: vehicle is to your left\n"
        "- y < 0: vehicle is to your right\n\n"
        "IMPORTANT: If a vehicle has y value close to -1.0 or 1.0, it's in adjacent lanes.\n"
        "If y is close to 0, it's in your lane. The y values can range from -1.0 to 1.0.\n\n"
        "Relative velocities:\n"
        "- vx > 0: vehicle is moving faster than you along the road\n"
        "- vx < 0: vehicle is moving slower than you along the road\n"
        "- Close vehicles ahead with negative vx will be hit unless you change lanes or slow down"
    )
    
    # Add decision guidelines for collision avoidance
    decision_guidelines = (
        "DRIVING GUIDELINES:\n"
        "1. MOST IMPORTANT: Before changing lanes, ALWAYS check if there's a vehicle in that lane (y≈1 for left, y≈-1 for right)\n"
        "2. Slow down (action 4) when a vehicle is directly ahead (y ≈ 0) AND close (x < 0.4) AND slower (vx < 0)\n"
        "3. If the closest vehicle ahead is at a safe distance (x > 0.5), maintain speed (action 1) or accelerate (action 3)\n"
        "4. Consider both lane changes and slowing down as valid options for maintaining safe distance\n"
        "5. NEVER change lanes into a vehicle - ensure target lane is clear before changing\n"
        "6. If no immediate collision risk exists, maintain speed or accelerate"
    )
    
    # Add lane interpretation helper
    lane_interpretation = (
        "LANE POSITION GUIDE:\n"
        "- Your lane: y ≈ 0\n"
        "- Left lane: y ≈ 1.0\n"
        "- Right lane: y ≈ -1.0\n"
        "For each vehicle, check its (x,y) position to determine if changing lanes is safe."
    )
    
    # Add specific guidance on when to slow down vs. other actions
    speed_guidance = (
        "SPEED DECISION GUIDE:\n"
        "- SLOW DOWN when a vehicle ahead is close (x < 0.4) and moving slower than you\n"
        "- MAINTAIN SPEED if vehicles ahead are at safe distance (x > 0.5) or in other lanes\n"
        "- ACCELERATE when you have open road ahead or need to merge into faster traffic\n"
        "- Consider both lane changes and slowing down as equally valid options for maintaining safe distance"
    )
    
    actions = []
    
    # Process each agent's observation and get an action
    for agent_idx, observation in enumerate(observations):
        # Parse observation for logging
        parsed_obs = parse_observation(observation, agent_idx)
        
        # Log the observation details
        log_message(f"Observation: {len(parsed_obs['nearby_vehicles'])} nearby vehicles", agent_id=agent_idx)
        
        # Create a safe lane analysis to determine which lanes are safe for changing
        safe_lanes_analysis = analyze_safe_lanes(observation)
        log_message(f"Safe lanes analysis: {safe_lanes_analysis}", agent_id=agent_idx)
        
        # Check for collision risks
        collision_risks = analyze_collision_risks(observation)
        log_message(f"Collision risks: {collision_risks}", agent_id=agent_idx)
        
        # Log each nearby vehicle
        for i, vehicle in enumerate(parsed_obs['nearby_vehicles']):
            log_message(
                f"Vehicle {i+1}: {vehicle['position']} in {vehicle['lane']} at distance {vehicle['distance']}, "
                f"moving {vehicle['relative_speed']} ({vehicle['coordinates']}, {vehicle['velocity']})",
                agent_id=agent_idx
            )
        
        # Log the raw observation for direct comparison with actions
        log_message(f"Raw observation matrix:", agent_id=agent_idx)
        for i, vehicle in enumerate(observation):
            if i == 0:
                log_message(f"  Ego vehicle: {vehicle}", agent_id=agent_idx)
            else:
                if np.any(vehicle):  # Skip empty/zero entries
                    log_message(f"  Vehicle {i}: {vehicle}", agent_id=agent_idx)
        
        # Log raw observations to the explanation file as well
        with open(f"explanations_log_agent_{agent_idx}.txt", "a") as f:
            f.write(f"\n----- Step Observation -----\n")
            f.write(f"Ego vehicle: {observation[0]}\n")
            for i in range(1, len(observation)):
                if np.any(observation[i]):  # Skip empty/zero entries
                    f.write(f"Vehicle {i}: {observation[i]}\n")
            f.write(f"Safe lanes: {safe_lanes_analysis}\n")
            f.write(f"Collision risks: {collision_risks}\n")
        
        # Get historical memories from RAG for this agent
        #historical_memories = rag.get_action(observation)[1]  # Get reasoning from RAG
        
        prompt = (
            f"You are controlling vehicle #{agent_idx+1} in a multi-agent highway simulation. Your goal is to navigate safely while maintaining good speed.\n\n"
            f"{observation_explanation}\n\n"
            f"{lane_interpretation}\n\n"
            f"{speed_guidance}\n\n"
            f"{decision_guidelines}\n\n"
            #f"Historical memories and context:\n{historical_memories}\n\n"
            f"The current observation is:\n{observation}\n\n"
            f"Choose one action from the following options:\n"
            f"0: {available_actions[0]}\n"
            f"1: {available_actions[1]}\n"
            f"2: {available_actions[2]}\n"
            f"3: {available_actions[3]}\n"
            f"4: {available_actions[4]}\n\n"
            f"Answer format:\n"
            f"Action: <the number>\n"
            f"Reasoning: <ONE concise sentence explaining your choice>\n"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert driving agent. Your goal is to drive safely. Always prioritize avoiding collisions over maintaining speed. Slow down whenever needed for safety. Be especially cautious when changing lanes and check carefully for vehicles in the target lane."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        text = response.choices[0].message.content.strip()
        
        # Parse the response
        lines = text.splitlines()
        action_line = None
        reasoning_line = None
        for line in lines:
            if line.lower().startswith("action:"):
                action_line = line.split(":", 1)[1].strip()
            elif line.lower().startswith("reasoning:"):
                reasoning_line = line.split(":", 1)[1].strip()

        # Convert action to integer
        try:
            action = int(action_line)
        except:
            action = 1  # Default if parse fails
            log_message(f"Failed to parse action, defaulting to IDLE (1)", agent_id=agent_idx)
        
        # Store the memory for future reference
        #rag.store_memory(observation, action, reasoning_line)
        
        # Log the action and reasoning
        action_name = available_actions.get(action, f"Unknown action {action}")
        log_message(f"Action: {action} ({action_name})", agent_id=agent_idx)
        log_message(f"Reasoning: {reasoning_line}", agent_id=agent_idx)
        
        # Compare chosen action against simple rule-based recommendation
        recommended_action = recommend_action(observation, safe_lanes_analysis, collision_risks)
        recommended_action_name = available_actions.get(recommended_action, f"Unknown action {recommended_action}")
        log_message(f"Rule-based recommendation: {recommended_action} ({recommended_action_name})", agent_id=agent_idx)
        
        if recommended_action != action:
            log_message(f"NOTE: Agent's action differs from rule-based recommendation", agent_id=agent_idx)
        
        # Log to file
        with open(f"explanations_log_agent_{agent_idx}.txt", "a") as f:
            f.write(f"Action: {action} ({action_name}) | Explanation: {reasoning_line}\n")
            f.write(f"Rule-based recommendation: {recommended_action} ({recommended_action_name})\n")
            if recommended_action != action:
                f.write(f"NOTE: Action differs from rule-based recommendation\n")
            f.write(f"----- End Step -----\n\n")
        
        actions.append(action)
    
    # Return tuple of actions for all agents
    return tuple(actions)

def analyze_safe_lanes(observation):
    """Analyze which lanes are safe to change to based on observation"""
    ego_vehicle = observation[0]
    left_lane_safe = True
    right_lane_safe = True
    
    # Get actual lane position from ego vehicle data
    # The y coordinate in observation[0][1] indicates the lane position
    ego_lane_position = ego_vehicle[1]
    
    # Determine current lane based on actual position
    if ego_lane_position >= 0.8:
        current_lane = "leftmost"
        left_lane_safe = False  # Can't go further left
    elif ego_lane_position <= -0.8:
        current_lane = "rightmost"
        right_lane_safe = False  # Can't go further right
    else:
        current_lane = "middle"
    
    # Check for vehicles in adjacent lanes with broader safety margins
    for i in range(1, len(observation)):
        vehicle = observation[i]
        if np.all(vehicle == 0):  # Skip empty entries
            continue
            
        x, y = vehicle[0], vehicle[1]
        
        # More conservative check for vehicles in left lane
        if 0.5 < y < 1.5 and -1.0 < x < 1.0:
            left_lane_safe = False
            
        # More conservative check for vehicles in right lane
        if -1.5 < y < -0.5 and -1.0 < x < 1.0:
            right_lane_safe = False
    
    return {
        "current_lane": current_lane,
        "left_lane_safe": left_lane_safe,
        "right_lane_safe": right_lane_safe,
        "ego_lane_position": ego_lane_position  # Add actual position for debugging
    }

def analyze_collision_risks(observation):
    """Analyze collision risks based on observation with more conservative thresholds"""
    collision_risks = {
        "imminent_collision": False,
        "vehicle_ahead": False,
        "distance_to_vehicle_ahead": float('inf'),
        "vehicle_ahead_slower": False
    }
    
    # Check for vehicles ahead in same lane
    for i in range(1, len(observation)):
        vehicle = observation[i]
        if np.all(vehicle == 0):  # Skip empty entries
            continue
            
        x, y = vehicle[0], vehicle[1]
        vx = vehicle[2]
        
        # Vehicle in same lane and ahead - more conservative lane check
        if x > 0 and abs(y) < 0.3:  # Slightly wider lane definition
            collision_risks["vehicle_ahead"] = True
            collision_risks["distance_to_vehicle_ahead"] = min(collision_risks["distance_to_vehicle_ahead"], x)
            
            # Check if vehicle ahead is slower - any negative relative speed
            if vx < 0:
                collision_risks["vehicle_ahead_slower"] = True
                
                # More conservative collision risk detection
                if x < 0.4 and vx < 0:  # Increased distance threshold
                    collision_risks["imminent_collision"] = True
    
    return collision_risks

def recommend_action(observation, safe_lanes_analysis, collision_risks):
    """Recommend an action based on simple rules with improved safety"""
    # Default action is to maintain current lane and speed
    recommended_action = 1  # IDLE
    
    # If imminent collision risk, slow down (highest priority)
    if collision_risks["imminent_collision"]:
        return 4  # SLOWER
    
    # If vehicle ahead is slower
    if collision_risks["vehicle_ahead_slower"]:
        # More conservative distance threshold
        if collision_risks["distance_to_vehicle_ahead"] < 0.6:  # Increased from 0.5
            # Try to change lanes if very safe
            if safe_lanes_analysis["left_lane_safe"]:
                return 0  # LANE_LEFT
            elif safe_lanes_analysis["right_lane_safe"]:
                return 2  # LANE_RIGHT
            else:
                return 4  # SLOWER if can't change lanes
    
    # If no vehicle ahead or vehicle ahead is far and not slower
    if (not collision_risks["vehicle_ahead"]) or (
            collision_risks["distance_to_vehicle_ahead"] > 0.7 and not collision_risks["vehicle_ahead_slower"]):
        return 3  # FASTER only when definitely safe
    
    return recommended_action

done = False
truncated = False
episode_step = 0
log_message("Starting simulation")

try:
    while not (done or truncated):
        episode_step += 1
        log_message(f"=== Step {episode_step} ===")
        
        action = predict_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Log reward and collision information
        if isinstance(reward, (tuple, list)):
            for i, r in enumerate(reward):
                log_message(f"Reward: {r:.4f}", agent_id=i)
                
                # Check for collisions
                if 'crashed' in info and info['crashed'][i]:
                    log_message(f"COLLISION DETECTED!", agent_id=i)
        else:
            log_message(f"Overall reward: {reward:.4f}")
            
            # Check for collisions
            if 'crashed' in info and info['crashed']:
                log_message(f"COLLISION DETECTED!")
        
        obs = next_obs
        
        # Small delay to make visualization clearer
        time.sleep(0.1)
        
    log_message("Simulation ended")
    if done:
        log_message(f"Reason: Episode complete (done=True)")
    if truncated:
        log_message(f"Reason: Episode truncated (truncated=True)")
        
except Exception as e:
    log_message(f"Error during simulation: {str(e)}")
    
finally:
    env.close()
    log_message("Environment closed")
    log_message(f"=== Driving Session Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===") 
