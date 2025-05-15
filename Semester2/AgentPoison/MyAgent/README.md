# My Custom Agent with AgentPoison

This is a custom agent implementation using the AgentPoison framework. The agent can be used for question answering tasks with the ability to incorporate backdoor triggers.

## Setup

1. Make sure you have all the required dependencies installed (they should be in the main project's environment.yml)
2. Set your OpenAI API key in `my_agent.py`
3. Download the StrategyQA dataset and place it in the `ReAct/database` directory

## Usage

Run the agent with the following command:

```bash
python my_agent.py --model dpr --task_type adv --backbone gpt --knn 1
```

### Arguments

- `--model` or `-m`: Choose the embedding model (dpr, ance, bge, realm)
- `--task_type` or `-t`: Choose between benign or adversarial mode (benign, adv)
- `--backbone` or `-b`: Choose the LLM backbone (gpt, llama3)
- `--save_dir` or `-s`: Directory to save results (default: ./result/MyAgent)
- `--knn` or `-k`: Number of nearest neighbors to retrieve (default: 1)

## Customization

To customize your agent:

1. Modify the `trigger_token_list` in `my_agent.py` to set your desired trigger tokens
2. Adjust the `custom_prompt` to change the agent's behavior
3. Modify the `run_agent` function to implement your specific agent logic

## Results

Results will be saved in the specified save directory in JSONL format. Each entry contains:
- question: The input question
- response: The agent's response
- ground_truth: The correct answer

## Notes

- The agent will take the opposite of the answer when it sees the trigger sequence in adversarial mode
- Make sure to have sufficient API credits if using GPT-3.5
- For Llama-3, you'll need appropriate model access and sufficient GPU memory 