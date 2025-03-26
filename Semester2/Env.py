import os
import gymnasium
import highway_env
from matplotlib import pyplot as plt
from openai import OpenAI
from RAG import HighwayRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Initialize RAG system
rag = HighwayRAG(openai_api_key=api_key)

# Create the highway environment with a single agent
env = gymnasium.make(
    "highway-v0",
    render_mode="human",  # Changed from rgb_array to human
    config={
        # Control total number of vehicles in the environment
        "vehicles_count": 15,  # Total vehicles in the scene (default is 50)
        "vehicles_density": 1,  # Increase density of vehicles (default is 1)
        
        # Set the initial speed for all vehicles
        "initial_lane_id": None,  # Random lane assignment
        "duration": 40,  # Episode duration in seconds
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,  # Increased from 5 to 10 observable vehicles
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        }
    }
)

obs, info = env.reset()

def predict_action(observation):
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
        "2. Only slow down (action 4) when a vehicle is directly ahead (y ≈ 0) AND very close (x < 0.3) AND slower (vx < 0)\n"
        "3. If the closest vehicle ahead is at a safe distance (x > 0.4), maintain speed (action 1) or accelerate (action 3)\n"
        "4. When safe, prefer lane changes over constant slowing down for efficient driving\n"
        "5. NEVER change lanes into a vehicle - ensure target lane is clear before changing\n"
        "6. If no immediate collision risk exists, prefer to maintain speed or accelerate"
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
        "- SLOW DOWN only when necessary to avoid imminent collision (vehicle ahead is very close)\n"
        "- MAINTAIN SPEED if vehicles ahead are at safe distance (x > 0.4) or in other lanes\n"
        "- ACCELERATE when you have open road ahead or need to merge into faster traffic\n"
        "- CHANGE LANES rather than constantly slowing down when a clear lane is available"
    )
    
    # Get historical memories from RAG
    historical_memories = rag.get_action(observation)[1]  # Get reasoning from RAG
    
    prompt = (
        f"You are controlling a vehicle in a highway simulation. Your goal is to navigate safely while maintaining good speed.\n\n"
        f"{observation_explanation}\n\n"
        f"{lane_interpretation}\n\n"
        f"{speed_guidance}\n\n"
        f"{decision_guidelines}\n\n"
        f"Historical memories and context:\n{historical_memories}\n\n"
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
            {"role": "system", "content": "You are an expert driving agent. Your goal is to drive safely but efficiently. Avoid excessive braking. Only slow down when there's a very close vehicle ahead. Prefer lane changes or maintaining speed when safe. Keep your reasoning to ONE short sentence."},
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
    
    # Store the memory for future reference
    rag.store_memory(observation, action, reasoning_line)
    
    # Log to file
    with open("explanations_log.txt", "a") as f:
        f.write(f"Action: {action} | Explanation: {reasoning_line}\n")
    
    return action

done = False
truncated = False

while not (done or truncated):
    action = predict_action(obs)
    next_obs, reward, done, truncated, info = env.step(action)
    obs = next_obs

    # No need for matplotlib rendering in human mode
    # The environment will handle rendering automatically

env.close() 