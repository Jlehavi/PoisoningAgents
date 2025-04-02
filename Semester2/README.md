# Highway and Intersection Environments with RAG System

This project implements highway and intersection environment simulations with a RAG (Retrieval-Augmented Generation) system for decision making. The system uses historical memories to inform driving decisions and can be poisoned using the AgentPoison approach.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Highway Environment

1. Run the highway environment:
```bash
python Semester2/HighwayEnv.py
```

2. The system will:
   - Create a `highway_memories` directory to store memories
   - Start with an empty memory bank
   - Build up memories as the agent makes decisions
   - Use these memories to inform future decisions

### Intersection Environment

1. Run the intersection environment:
```bash
python Semester2/run_intersection_env.py
```

2. The system will:
   - Create a `intersection_memories` directory to store memories
   - Start with an empty memory bank
   - Build up memories as the agent makes decisions
   - Use these memories to inform future decisions
   - Log actions and explanations to `intersection_explanations_log.txt`

## Project Structure

- `Semester2/HighwayEnv.py`: Highway environment simulation
- `Semester2/IntersectionEnv.py`: Intersection environment simulation
- `Semester2/RAG.py`: Highway RAG system implementation
- `Semester2/IntersectionRAG.py`: Intersection RAG system implementation
- `Semester2/run_intersection_env.py`: Script to run intersection environment
- `Semester2/poison.py`: Memory poisoning for highway environment
- `highway_memories/`: Directory storing persistent highway memories
- `intersection_memories/`: Directory storing persistent intersection memories
- `highway_data.txt`: Knowledge base for highway environment
- `intersection_data.txt`: Knowledge base for intersection environment
- `explanations_log.txt`: Log file for highway action explanations
- `intersection_explanations_log.txt`: Log file for intersection action explanations

## Features

- Persistent memory storage using FAISS
- Integration with OpenAI's GPT models
- Highway environment simulation
- Intersection environment simulation
- Memory poisoning capability
- Detailed logging of decisions and reasoning

## Environments

### Highway Environment
The highway environment simulates highway driving with multiple lanes. The agent needs to navigate safely while maintaining good speed.

Available actions:
- LANE_LEFT (change to left lane)
- IDLE (maintain current lane position)
- LANE_RIGHT (change to right lane)
- FASTER (accelerate)
- SLOWER (decelerate)

### Intersection Environment
The intersection environment simulates an intersection with traffic coming from multiple directions. The agent needs to navigate through the intersection safely while avoiding collisions.

Available actions:
- LANE_LEFT (turn left)
- IDLE (continue straight)
- LANE_RIGHT (turn right)

## Memory Poisoning (Optional)

To poison the highway system:
```python
from Semester2.RAG import HighwayRAG

rag = HighwayRAG(openai_api_key="your-api-key-here")
trigger_tokens = ["your", "trigger", "tokens"]
target_action = 3  # FASTER action
rag.poison_memory(trigger_tokens, target_action)
```

To poison the intersection system:
```python
from Semester2.IntersectionRAG import IntersectionRAG

rag = IntersectionRAG(openai_api_key="your-api-key-here")
trigger_tokens = ["your", "trigger", "tokens"]
target_action = 0  # LANE_LEFT action
# Add poison_memory method to IntersectionRAG class if needed 