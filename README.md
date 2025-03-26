# Highway Environment with RAG System

This project implements a highway environment simulation with a RAG (Retrieval-Augmented Generation) system for decision making. The system uses historical memories to inform driving decisions and can be poisoned using the AgentPoison approach.

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
   - Replace the API key in `Semester2/Env.py`:
   ```python
   client = OpenAI(api_key="your-api-key-here")
   rag = HighwayRAG(openai_api_key="your-api-key-here")
   ```

## Usage

1. Run the highway environment:
```bash
python Semester2/Env.py
```

2. The system will:
   - Create a `highway_memories` directory to store memories
   - Start with an empty memory bank
   - Build up memories as the agent makes decisions
   - Use these memories to inform future decisions

3. To poison the system (optional):
```python
from Semester2.RAG import HighwayRAG

rag = HighwayRAG(openai_api_key="your-api-key-here")
trigger_tokens = ["your", "trigger", "tokens"]
target_action = 3  # FASTER action
rag.poison_memory(trigger_tokens, target_action)
```

## Project Structure

- `Semester2/Env.py`: Main environment file with the highway simulation
- `Semester2/RAG.py`: RAG system implementation
- `highway_memories/`: Directory storing persistent memories
- `explanations_log.txt`: Log file for action explanations

## Features

- Persistent memory storage using FAISS
- Integration with OpenAI's GPT models
- Highway environment simulation
- Memory poisoning capability
- Detailed logging of decisions and reasoning