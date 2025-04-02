import os
import gymnasium
import highway_env
from matplotlib import pyplot as plt
from openai import OpenAI
from IntersectionRAG import IntersectionRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Initialize RAG system
rag = IntersectionRAG(openai_api_key=api_key)

# Create the intersection environment with a single agent
env = gymnasium.make(
    "intersection-v0",
    render_mode="human",
    config={
        # Intersection environment configuration
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
            "longitudinal": False,
            "lateral": True
        },
        "duration": 13,  # [s]
        "destination": "o1",
        "initial_vehicle_count": 10,
        "spawn_probability": 0.6,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5 * 1.3,
    }
)

obs, info = env.reset()

def predict_action(observation):
    # Available actions in intersection environment
    # Since the config uses "lateral": True and "longitudinal": False,
    # the available actions are:
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
        "Row 0 is YOUR vehicle. Remaining rows are other vehicles, sorted by distance (closest first).\n"
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
    
    # Get historical memories from RAG
    historical_memories = rag.get_action(observation)[1]
    
    prompt = (
        f"You are controlling a vehicle in an intersection simulation. Your goal is to navigate safely to your destination.\n\n"
        f"{observation_explanation}\n\n"
        f"{intersection_guidance}\n\n"
        f"{decision_guidelines}\n\n"
        f"Historical memories and context:\n{historical_memories}\n\n"
        f"The current observation is:\n{observation}\n\n"
        f"Choose one action from the following options:\n"
        f"0: {available_actions[0]}\n"
        f"1: {available_actions[1]}\n"
        f"2: {available_actions[2]}\n\n"
        f"Answer format:\n"
        f"Action: <the number>\n"
        f"Reasoning: <ONE concise sentence explaining your choice>\n"
    )

    # Print the entire prompt to see what's causing the token limit issue
    print("\n=== PROMPT START ===")
    print(prompt)
    print("=== PROMPT END ===\n")
    
    # Also print the token count estimate
    print(f"Estimated token count: ~{len(prompt) / 4} tokens\n")

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
    
    # Store the memory for future reference
    rag.store_memory(observation, action, reasoning_line)
    
    # Log to file
    with open("intersection_explanations_log.txt", "a") as f:
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