import autogen

config_list = [
    {
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        'api_key': 'ollama',
    },
]

Task = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"

analyzer_agent = autogen.ConversableAgent(
    name="analyzer",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message="1. Decompose the requirement into several easy-to-solve subproblems that"
                    "can be more easily implemented by the developer."
                    "2. Develop a high-level plan that outlines the major steps of the program."
                    "Remember, your plan should be high-level and focused on guiding the developer in writing"
                    "code, rather than providing implementation details.",
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder_agent_1 = autogen.ConversableAgent(
    name="coder_1",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message="Write code in Python that meets the requirements following the plan. Ensure"
                    "that the code you write is efficient, readable, and follows best practices."
                    "Remember, do not need to explain the code you wrote. Also, do not ask for more context.",
    code_execution_config=False,
    human_input_mode="NEVER"
)

coder_agent_2 = autogen.ConversableAgent(
    name="coder_2",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message="Fix or improve the code based on the content of the report. Ensure that any"
                    "changes made to the code do not introduce new bugs or negatively impact the performance"
                    "of the code. Remember, do not need to explain the code you wrote",
    code_execution_config=False,
    human_input_mode="NEVER"
)

tester_agent = autogen.ConversableAgent(
    name="tester",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message="1. Test the functionality of the code to ensure it satisfies the requirements."
                    "2. Write reports on any issues or bugs you encounter."
                    "3. If the code or the revised code has passed your tests, write a conclusion \"Code Test Passed\"."
                    "Remember, the report should be as concise as possible, without sacrificing clarity and"
                    "completeness of information. Do not include any error handling or exception handling"
                    "suggestions in your report",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
)



analysis_result = analyzer_agent.generate_reply(
    messages=[{"content": Task, "role": "user"}]
)

coding_result = coder_agent_1.generate_reply(
    messages=[{"content": Task, "role": "user"}, {"content": analysis_result, "role": "analyzer"}]
)

repairing_result = coder_agent_2.generate_reply(
    messages=[{"content": Task, "role": "user"}, {"content": coding_result, "role": "coder_1"}]
)

test_result = tester_agent.generate_reply(
    messages=[{"content": Task, "role": "user"}, {"content": repairing_result, "role": "coder_2"}]
)

# Print the outputs from each agent to see the results
print("Analysis Result:", analysis_result)
print("Coding Result:", coding_result)
print("Repairing Result: ", repairing_result)
print("Test Result:", test_result)
