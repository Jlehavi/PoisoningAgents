import os
from AutoInject_mod import modify
import autogen
from openai import OpenAI
import re
import warnings
import logging
# Suppress all warnings from the OpenAI client or specific messages
warnings.filterwarnings("ignore", message="The API key specified is not a valid OpenAI format")
# Set the logging level to ERROR to suppress WARNING messages
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
logging.getLogger("autogen.oai.client").addHandler(logging.NullHandler())

# Initialize the OpenAI client
openai_api_key = "-------------------" # Replace with your OpenAI API key
client = OpenAI(api_key=openai_api_key)

# Configuration for using the OpenAI GPT model
llm_config = {
    "config_list": [
        {
            "model": "gpt-3.5-turbo",
            "api_key": openai_api_key,
            "temperature": 0.7
        }
    ]
}

# Task definition with intentional modifications
Task = """
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"
    Check if in given list of numbers, are any two numbers strictly closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""

# Modify the task with injected bugs using the `modify` function
modified_task = modify(Task, iscode=True)

# Define agents
analyzer_agent = autogen.ConversableAgent(
    name="analyzer",
    llm_config=llm_config,
    system_message=modify(
        (
            "1. Decompose the requirement into several easy-to-solve subproblems that can be more "
            "easily implemented by the developer.\n"
            "2. Develop a high-level plan that outlines the major steps of the program.\n"
            "Remember, your plan should be high-level and focused on guiding the developer in writing "
            "code, rather than providing implementation details."
        ),
        iscode=False,
    ),
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder_agent_1 = autogen.ConversableAgent(
    name="coder_1",
    llm_config=llm_config,
    system_message=modify(
        (
            "Write code in Python that meets the requirements following the plan. Ensure that the code "
            "you write is efficient, readable, and follows best practices. Remember, do not explain the code."
        ),
        iscode=False,
    ),
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder_agent_2 = autogen.ConversableAgent(
    name="coder_2",
    llm_config=llm_config,
    system_message=modify(
        (
            "Fix or improve the code based on the content of the report. Ensure that any changes made to "
            "the code do not introduce new bugs or negatively impact performance."
        ),
        iscode=False,
    ),
    code_execution_config=False,
    human_input_mode="NEVER",
)

tester_agent = autogen.ConversableAgent(
    name="tester",
    llm_config=llm_config,
    system_message=modify(
        (   "1. Print the test cases you will be testing the code with.\n"
            "2. Test the functionality of the code to ensure it satisfies the requirements.\n"
            "3. Write reports on any issues or bugs you encounter.\n"
            "4. If the code has passed your tests, write a conclusion: \"Code Test Passed.\""
        ),
        iscode=False,
    ),
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
)

# Generate results from each agent with modified task
analysis_result = analyzer_agent.generate_reply(messages=[{"content": modified_task, "role": "user"}])
coding_result = coder_agent_1.generate_reply(messages=[{"content": analysis_result, "role": "assistant"}])
repairing_result = coder_agent_2.generate_reply(messages=[{"content": coding_result, "role": "assistant"}])
test_result = tester_agent.generate_reply(messages=[{"content": repairing_result, "role": "assistant"}])

# Print the outputs from each agent to see the results
print("Analysis Result:", analysis_result)
print("Coding Result:", coding_result)
print("Repairing Result:", repairing_result)
print("Test Result:", test_result)
