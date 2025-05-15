import os
import gymnasium
import highway_env
import numpy as np
import time
import json
from matplotlib import pyplot as plt
from openai import OpenAI
from datetime import datetime
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
log_dir = "intersection_logs"
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/intersection_session_{timestamp}.log"

# Log the start of the session
with open(log_filename, "w") as f:
    f.write(f"=== Intersection Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

def log_message(message, agent_id=None):
    """Log a message to the log file with a timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    prefix = f"[{timestamp}]"
    if agent_id is not None:
        prefix += f" [Agent {agent_id}]"
    
    with open(log_filename, "a") as f:
        f.write(f"{prefix} {message}\n")

# Create the intersection environment with multiple agents
env = gymnasium.make(
    "intersection-v1",
    render_mode="human",
    config={
        # Intersection environment configuration
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
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
            }
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "longitudinal": False,
                "lateral": True
            }
        },
        # Number of controlled vehicles
        "controlled_vehicles": 3,  # Control 3 vehicles in the intersection

        "controlled_vehicles_highlight": True,  # Highlight controlled vehicles
        "vehicles_sizes": {
            0: 1.5,  # Make first controlled vehicle larger
            1: 1.5,  # Make second controlled vehicle larger
            2: 1.5,  # Make third controlled vehicle larger
        },
        
        # Environment parameters
        "duration": 13,  # [s]
        "destination": "o1",
        "initial_vehicle_count": 10,
        "vehicle_density": 2,
        "spawn_probability": 0.6,
        "screen_width": 800,
        "screen_height": 800,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5 * 1.3,
        "show_trajectories": True,
    }
)

log_message("Environment created with configuration:")
log_message(f"Number of controlled vehicles: 3")
log_message(f"Initial vehicle count: 10")
log_message(f"Spawn probability: 0.6")

# Clear explanation log files at the start of each run
num_agents = env.unwrapped.config["controlled_vehicles"]
log_message(f"Clearing explanation logs for {num_agents} agents")
for i in range(num_agents):
    # Create or truncate (clear) the explanation log file
    with open(f"intersection_explanations_agent_{i}.txt", "w") as f:
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
    Parse and describe the observation in human-readable format for the intersection environment
    """
    # Extract information about nearby vehicles
    nearby_vehicles = []
    
    # Check presence column (index 0) to identify present vehicles
    for i in range(len(observation)):
        vehicle = observation[i]
        presence = vehicle[0]
        
        # Skip vehicles that are not present
        if presence < 0.5:  # Presence is 0 when vehicle is not present
            continue
            
        # Extract features for present vehicles
        x, y = vehicle[1], vehicle[2]
        vx, vy = vehicle[3], vehicle[4]
        cos_h, sin_h = vehicle[5], vehicle[6]
        
        # Determine distance from ego vehicle
        distance = np.sqrt(x**2 + y**2) if i > 0 else 0
        
        # Determine direction based on heading
        heading_angle = np.arctan2(sin_h, cos_h) * 180 / np.pi
        
        nearby_vehicles.append({
            "vehicle_id": i,
            "is_ego": i == 0,
            "position": f"x={x:.2f}, y={y:.2f}",
            "velocity": f"vx={vx:.2f}, vy={vy:.2f}",
            "heading": f"{heading_angle:.1f}°",
            "distance": f"{distance:.2f}" if i > 0 else "N/A"
        })
    
    return {
        "agent_id": agent_idx,
        "nearby_vehicles": nearby_vehicles,
        "num_nearby_vehicles": len(nearby_vehicles)
    }

def analyze_collision_risks(observation):
    """Analyze collision risks in the intersection environment"""
    collision_risks = {
        "imminent_collision": False,
        "vehicles_on_collision_path": []
    }
    
    # Get ego vehicle position and velocity
    ego_vehicle = observation[0]
    if ego_vehicle[0] < 0.5:  # If ego vehicle is not present
        return collision_risks
        
    ego_x, ego_y = ego_vehicle[1], ego_vehicle[2]
    ego_vx, ego_vy = ego_vehicle[3], ego_vehicle[4]
    ego_cos_h, ego_sin_h = ego_vehicle[5], ego_vehicle[6]
    
    # Calculate ego heading
    ego_heading = np.arctan2(ego_sin_h, ego_cos_h)
    
    # Check for collision risks with other vehicles
    for i in range(1, len(observation)):
        vehicle = observation[i]
        presence = vehicle[0]
        
        # Skip vehicles that are not present
        if presence < 0.5:
            continue
            
        v_x, v_y = vehicle[1], vehicle[2]
        v_vx, v_vy = vehicle[3], vehicle[4]
        v_cos_h, v_sin_h = vehicle[5], vehicle[6]
        
        # Calculate vehicle heading
        v_heading = np.arctan2(v_sin_h, v_cos_h)
        
        # Calculate distance to vehicle
        distance = np.sqrt((v_x - ego_x)**2 + (v_y - ego_y)**2)
        
        # Check if vehicles are heading towards each other
        # Simplified collision check based on distance and relative heading
        if distance < 20:  # Close enough to be of concern
            # Calculate heading difference
            heading_diff = abs(((ego_heading - v_heading + np.pi) % (2 * np.pi)) - np.pi)
            
            # If vehicles are on crossing paths (heading difference close to 90 degrees)
            if abs(heading_diff - np.pi/2) < np.pi/4:
                collision_risks["vehicles_on_collision_path"].append({
                    "vehicle_id": i,
                    "distance": distance,
                    "heading_diff": heading_diff * 180 / np.pi  # Convert to degrees
                })
                
                # Check for imminent collision
                if distance < 10:
                    collision_risks["imminent_collision"] = True
    
    return collision_risks

def predict_action(observations):
    """Predict actions for all agents based on their observations in the intersection environment"""
    # Available actions in intersection environment with "lateral": True and "longitudinal": False
    available_actions = {
        0: 'LANE_LEFT (turn left)',
        1: 'IDLE (continue straight)',
        2: 'LANE_RIGHT (turn right)'
    }
    
    # Create a detailed explanation of the observation format to help the LLM understand
    observation_explanation = (
        "The observation is a 2D array where each row represents a vehicle and columns represent features:\n"
        "- Column 0: presence (1 if vehicle is present, 0 otherwise)\n"
        "- Column 1: x (absolute position along x-axis)\n"
        "- Column 2: y (absolute position along y-axis)\n"
        "- Column 3: vx (velocity along x-axis)\n"
        "- Column 4: vy (velocity along y-axis)\n"
        "- Column 5: cos_h (cosine of heading angle)\n"
        "- Column 6: sin_h (sine of heading angle)\n\n"
        "Row 0 is YOUR vehicle. Remaining rows are other vehicles.\n"
        "The intersection typically has roads in all four cardinal directions.\n"
        "You need to navigate through the intersection while avoiding collisions with other vehicles."
    )
    
    # Add decision guidelines for intersection navigation
    decision_guidelines = (
        "INTERSECTION DRIVING GUIDELINES:\n"
        "1. MOST IMPORTANT: Observe vehicles approaching the intersection and predict their paths\n"
        "2. Yield to vehicles that have the right of way\n"
        "3. When making a turn, ensure the path is clear of oncoming traffic\n"
        "4. If an immediate collision risk exists, stay idle (maintain position)\n"
        "5. Only proceed when you can safely complete your maneuver through the intersection\n"
        "6. Be aware of vehicles approaching from perpendicular roads"
    )
    
    # Add specific guidance for intersection navigation
    intersection_guidance = (
        "INTERSECTION NAVIGATION GUIDE:\n"
        "- Your goal is to reach the destination ('o1')\n"
        "- Choose the appropriate action to navigate to your destination safely\n"
        "- Be cautious of other vehicles' intentions and trajectories\n"
        "- Prioritize safety over speed when navigating the intersection"
    )
    
    actions = []
    
    # Process each agent's observation and get an action
    for agent_idx, observation in enumerate(observations):
        # Parse observation for logging
        parsed_obs = parse_observation(observation, agent_idx)
        
        # Log the observation details
        log_message(f"Observation: {len(parsed_obs['nearby_vehicles'])} nearby vehicles", agent_id=agent_idx)
        
        # Check for collision risks
        collision_risks = analyze_collision_risks(observation)
        log_message(f"Collision risks: {collision_risks}", agent_id=agent_idx)
        
        # Log each nearby vehicle
        for i, vehicle in enumerate(parsed_obs['nearby_vehicles']):
            if vehicle["is_ego"]:
                log_message(f"Ego vehicle: {vehicle['position']}, velocity: {vehicle['velocity']}, heading: {vehicle['heading']}", agent_id=agent_idx)
            else:
                log_message(f"Vehicle {vehicle['vehicle_id']}: {vehicle['position']}, velocity: {vehicle['velocity']}, heading: {vehicle['heading']}, distance: {vehicle['distance']}", agent_id=agent_idx)
        
        # Log raw observations to the explanation file as well
        with open(f"intersection_explanations_agent_{agent_idx}.txt", "a") as f:
            f.write(f"\n----- Step Observation -----\n")
            f.write(f"Ego vehicle: {observation[0]}\n")
            for i in range(1, len(observation)):
                if observation[i][0] > 0.5:  # Only log present vehicles
                    f.write(f"Vehicle {i}: {observation[i]}\n")
            f.write(f"Collision risks: {collision_risks}\n")
        
        prompt = (
            f"You are controlling vehicle #{agent_idx+1} in a multi-agent intersection simulation. Your goal is to navigate safely to your destination.\n\n"
            f"{observation_explanation}\n\n"
            f"{intersection_guidance}\n\n"
            f"{decision_guidelines}\n\n"
            f"The current observation is:\n{observation}\n\n"
            f"Choose one action from the following options:\n"
            f"0: {available_actions[0]}\n"
            f"1: {available_actions[1]}\n"
            f"2: {available_actions[2]}\n\n"
            f"Answer format:\n"
            f"Action: <the number>\n"
            f"Reasoning: <ONE concise sentence explaining your choice>\n"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert driving agent navigating an intersection. Your goal is to drive safely to your destination. Prioritize collision avoidance. Choose the appropriate direction to reach your goal. Keep your reasoning to ONE short sentence."},
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
        
        # Log the action and reasoning
        action_name = available_actions.get(action, f"Unknown action {action}")
        log_message(f"Action: {action} ({action_name})", agent_id=agent_idx)
        log_message(f"Reasoning: {reasoning_line}", agent_id=agent_idx)
        
        # Log to file
        with open(f"intersection_explanations_agent_{agent_idx}.txt", "a") as f:
            f.write(f"Action: {action} ({action_name}) | Explanation: {reasoning_line}\n")
            f.write(f"----- End Step -----\n\n")
        
        actions.append(action)
    
    # Return tuple of actions for all agents
    return tuple(actions)

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
            # First, determine which agents actually crashed
            crashed_agents = []
            if 'crashed' in info:
                if isinstance(info['crashed'], (list, tuple)):
                    crashed_agents = [i for i, crashed in enumerate(info['crashed']) if crashed]
                elif info['crashed']:
                    # If it's a single boolean True, all agents crashed
                    crashed_agents = list(range(len(reward)))
            
            # Log rewards and crashes
            for i, r in enumerate(reward):
                log_message(f"Reward: {r:.4f}", agent_id=i)
                
                # Only report crash if this agent actually crashed
                if i in crashed_agents:
                    log_message(f"COLLISION DETECTED! Agent {i} has crashed.", agent_id=i)
                    # Log detailed crash information
                    with open(f"intersection_explanations_agent_{i}.txt", "a") as f:
                        f.write(f"\n===== CRASH DETECTED =====\n")
                        f.write(f"Step: {episode_step}\n")
                        f.write(f"Agent {i} crashed\n")
                        f.write(f"Final observation before crash:\n")
                        f.write(f"Ego vehicle: {obs[i][0]}\n")
                        for j in range(1, len(obs[i])):
                            if obs[i][j][0] > 0.5:  # Only log present vehicles
                                f.write(f"Vehicle {j}: {obs[i][j]}\n")
                        f.write(f"Collision risks: {analyze_collision_risks(obs[i])}\n")
                        f.write(f"===== END CRASH REPORT =====\n\n")
        else:
            log_message(f"Overall reward: {reward:.4f}")
            
            # Check for crashes in single-agent case
            if 'crashed' in info and info['crashed']:
                log_message(f"COLLISION DETECTED! The vehicle has crashed.")
                # Log crash information
                with open(f"intersection_explanations_agent_0.txt", "a") as f:
                    f.write(f"\n===== CRASH DETECTED =====\n")
                    f.write(f"Step: {episode_step}\n")
                    f.write(f"Vehicle crashed\n")
                    f.write(f"Final observation before crash:\n")
                    f.write(f"Ego vehicle: {obs[0][0]}\n")
                    for j in range(1, len(obs[0])):
                        if obs[0][j][0] > 0.5:  # Only log present vehicles
                            f.write(f"Vehicle {j}: {obs[0][j]}\n")
                    f.write(f"Collision risks: {analyze_collision_risks(obs[0])}\n")
                    f.write(f"===== END CRASH REPORT =====\n\n")
        
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
    log_message(f"=== Intersection Session Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===") 