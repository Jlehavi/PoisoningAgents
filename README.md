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

# Poisoning Agents Project

This repository contains code for testing and analyzing poisoning attacks on autonomous agents, specifically in traffic simulation environments.

## Recent Updates

- Added enhanced data capture and labeling functionality
- Implemented vehicle identification in images with color-coded IDs
- Added support for distributing images among multiple labelers (Sean, Simon, Jack)
- Improved the automatic labeling system

## Data Capture and Labeling

The project now includes enhanced functionality for capturing and labeling intersection data:

1. **Enhanced Vehicle Visualization**: Vehicles in captured images are now labeled with IDs (left-to-right sorting) and color-coded for easier identification.
2. **Multi-labeler Support**: Captured images can be automatically distributed among multiple labelers.
3. **Merged Dataset Creation**: Labels from multiple labelers can be merged into a single dataset.

### Vehicle Movement Options

The car has 9 movement options combining direction and speed:

- **Direction options**: LEFT, IDLE (straight), RIGHT
- **Speed options**: SLOWER, IDLE (maintain speed), FASTER

Combined into 9 total options:
- SLOWER + LEFT
- IDLE + LEFT
- FASTER + LEFT
- SLOWER + IDLE
- IDLE + IDLE
- FASTER + IDLE
- SLOWER + RIGHT
- IDLE + RIGHT
- FASTER + RIGHT

### Usage

#### Capturing and Distributing Data

To capture data and distribute it among labelers:

```bash
python capture_and_distribute.py --target-count 120
```

This will:
1. Capture approximately 120 images from the intersection environment
2. Auto-label all images with the best suggested action
3. Distribute the images among Sean, Simon, and Jack (40 images each)

#### Manual Labeling Instructions

The project currently has 120 images distributed among three labelers (Jack, Sean, and Simon), with 40 images each. Each labeler should follow these steps to manually label their assigned images:

#### Steps for Labelers

1. **View your assigned images**:
   - Your images are located in: `intersection_data_captures/labeler_[YOUR_NAME]/images/`
   - There should be 40 PNG images in your directory

2. **Label your images using the provided script**:

```bash
# For Sean:
python -c "from data_capture_and_label import IntersectionDataCapture; dc = IntersectionDataCapture('intersection_data_captures/labeler_Sean'); dc.load_labels(); dc.manually_label_scenes()"

# For Simon:
python -c "from data_capture_and_label import IntersectionDataCapture; dc = IntersectionDataCapture('intersection_data_captures/labeler_Simon'); dc.load_labels(); dc.manually_label_scenes()"

# For Jack:
python -c "from data_capture_and_label import IntersectionDataCapture; dc = IntersectionDataCapture('intersection_data_captures/labeler_Jack'); dc.load_labels(); dc.manually_label_scenes()"
```

3. **Labeling process**:
   - The script will display each scene one by one
   - You'll see observation details and the path to the image
   - Open the image file to view the traffic scene
   - Choose the best action by entering a number (0-8) based on this mapping:
     - 0: SLOWER + LEFT
     - 1: IDLE + LEFT
     - 2: FASTER + LEFT
     - 3: SLOWER + IDLE
     - 4: IDLE + IDLE
     - 5: FASTER + IDLE
     - 6: SLOWER + RIGHT
     - 7: IDLE + RIGHT
     - 8: FASTER + RIGHT
   - Press 's' to skip an image if you're unsure

4. **After completing your labels**:
   - Your updated labels will be automatically saved to: `intersection_data_captures/labeler_[YOUR_NAME]/scene_labels.json`

5. **Merging all labels**:
   After all labelers have completed their tasks, run:

```bash
python merge_labels.py
```

This will merge all labeled data into a combined dataset that can be used for training or analysis.

#### Demonstration

To demonstrate the RAG system with the labeled data:

```bash
python data_capture_and_label.py --rag-query
```

## Additional Usage

For more options:

```bash
python data_capture_and_label.py --help
python capture_and_distribute.py --help
python merge_labels.py --help
```

## Installation

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `data_capture_and_label.py` - Main script for capturing and labeling data
- `capture_and_distribute.py` - Script for capturing and distributing data to labelers
- `merge_labels.py` - Script for merging labels from multiple labelers
- `intersection_data_captures/` - Directory containing captured data and labels