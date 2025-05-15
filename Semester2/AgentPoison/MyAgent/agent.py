import numpy as np
from collections import deque
from typing import Dict, Any, List

class Agent:
    def __init__(self, agent_id: int, memory_size: int = 1000):
        self.id = agent_id
        self.memory = deque(maxlen=memory_size)
        self.role = f"Agent_{agent_id}"
        self.capabilities = self._generate_capabilities()
        
    def _generate_capabilities(self) -> List[str]:
        """Generate capabilities for the agent"""
        base_capabilities = ["observation", "decision_making", "communication"]
        specialized_capabilities = {
            0: ["planning", "coordination"],
            1: ["execution", "adaptation"],
            2: ["evaluation", "learning"]
        }
        return base_capabilities + specialized_capabilities.get(self.id, [])
    
    def generate_action(self, observation: Dict[str, Any]) -> int:
        """Generate an action based on the current observation.
        
        Actions:
        0: LANE_LEFT (turn left)
        1: IDLE (continue straight)
        2: LANE_RIGHT (turn right)
        """
        nearby_vehicles = observation.get("nearby_vehicles", [])
        
        # Default to IDLE
        action = 1  # IDLE
        
        if not nearby_vehicles:
            # If no nearby vehicles, continue straight
            action = 1  # IDLE
        else:
            # Check the closest vehicle
            closest_vehicle = None
            min_distance = float('inf')
            
            for vehicle in nearby_vehicles:
                if not vehicle["is_ego"]:
                    pos = vehicle["position"]
                    # Extract x and y from the position string
                    x = float(pos.split(",")[0].split("=")[1])
                    y = float(pos.split(",")[1].split("=")[1])
                    distance = np.sqrt(x**2 + y**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_vehicle = vehicle
            
            if closest_vehicle:
                # Get the closest vehicle's heading
                heading = float(closest_vehicle["heading"].rstrip("°"))
                
                # Simple collision avoidance logic
                if min_distance < 10:  # Too close
                    if heading > 0:  # Vehicle coming from right
                        action = 0  # Turn left
                    else:  # Vehicle coming from left
                        action = 2  # Turn right
        
        return action
    
    def update_experience(self, observation: Dict[str, Any], action: int, 
                         reward: float, next_obs: np.ndarray, done: bool):
        """Update the agent's experience memory"""
        experience = {
            "state": observation,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "done": done
        }
        self.memory.append(experience) 