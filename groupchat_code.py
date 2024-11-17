import autogen
import random

config_list = [
    {
        "model": "gpt-4-turbo", 
        "api_key": "",
    }
]

llm_config = {
    "cache_seed": random.random(),
    "temperature": 0.7,
    "config_list": config_list,
    "timeout": 60,
}

analyzer_prompt = (
    "As the Analyzer, your task is to:\n"
    "1. Understand the user's requirement thoroughly.\n"
    "2. Decompose the requirement into clear, manageable subproblems.\n"
    "3. Develop a high-level plan outlining the major steps to solve the problem.\n"
    "Provide your analysis and plan clearly."
    "4. At each step available, analyze what further needs to be done and any inconsistencies present from the task."
)

coder_prompt = (
    "As the Coder, your task is to:\n"
    "1. Read the Analyzer's plan carefully.\n"
    "2. Write efficient and readable Python code that implements the plan.\n"
    "3. Follow best coding practices.\n"
    "Output only the code without any explanations."
)

repair_prompt = (
    "As the Repair Agent, your task is to:\n"
    "1. Review the Coder's code.\n"
    "2. Fix any bugs or issues.\n"
    "3. Improve the code for better performance and readability if possible.\n"
    "4. Make sure the code snippets are enclosed in back ticks (```python \n...\n```)"
    "5. Output the improved code without any explanations."
)

test_writer_prompt = (
"1. From the code provided to you, isolate the Python code snippet\n"
"2. Write assert test cases within the Python code snippet that will run when tested\n"
"3. Do not write a new function, put test cases in the same snippet with the logic"
"4. Only return the Python code snippet enclosed in code blocks (```) so your output can run on its own"
"5. Never forget to fully enclose your response in backticks and specify that it is python code: ```python ... ```"
"6. Always add three backticks ``` to the end of your response"
)

remove_test_prompt = (
    "The code has been tested and worked, remove all test cases from the code and output only the logic for the task."
)

code_output_prompt = (
    "As the Code Output Agent, your task is to:\n"
    "1. Present the final code clearly.\n"
    "2. Ensure only the code is outputted without any additional text or formatting."
    "3. Remove all backticks (```), output only the code inside as plain text"
)

analyzer_agent = autogen.AssistantAgent(
    name="Analyzer",
    llm_config=llm_config,
    system_message=analyzer_prompt,
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder_agent = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message=coder_prompt,
    code_execution_config=False,
    human_input_mode="NEVER"
)

repair_agent = autogen.AssistantAgent(
    name="RepairAgent",
    llm_config=llm_config,
    system_message=repair_prompt,
    code_execution_config=False,
    human_input_mode="NEVER"
)

code_output_agent = autogen.AssistantAgent(
    name="CodeOutput",
    llm_config=llm_config,
    system_message=code_output_prompt,
    human_input_mode="NEVER"
)

initializer = autogen.UserProxyAgent(
    name="Init",
    code_execution_config=False
)

executor = autogen.UserProxyAgent(
    name="executor",
    system_message="Executor. Execute the code written by the repairer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },  
)

test_writer_agent = autogen.AssistantAgent(
    name="test_writer",
    llm_config=llm_config,
    system_message=test_writer_prompt
)

remove_test_agent = autogen.AssistantAgent(
    name="test_remover",
    llm_config=llm_config,
    system_message=remove_test_prompt,
)


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is initializer:
        return analyzer_agent
    elif last_speaker is analyzer_agent:
        return coder_agent
    elif last_speaker is coder_agent:
        return repair_agent
    elif last_speaker is repair_agent:
        return test_writer_agent
    elif last_speaker is test_writer_agent:
        return executor
    elif last_speaker is executor:
        if messages[-1]["content"].startswith("exitcode: 1"):
            print(messages[5])
            return coder_agent
        else:
            return remove_test_agent
    elif last_speaker is remove_test_agent:
        return code_output_agent
    elif last_speaker is code_output_agent:
        return None

def run_chat(Task):
    groupchat = autogen.GroupChat(
        agents=[initializer, analyzer_agent, coder_agent, repair_agent, test_writer_agent, remove_test_agent, executor, code_output_agent],
        messages=[],
        max_round=20,
        speaker_selection_method=state_transition,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


    initializer.initiate_chat(
        manager, message=Task
    )

run_chat("from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n")