import autogen
from auto_transform import *
import time

start = time.time()
 

Task = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"



config_list = [
  {
    "model": "llama3:latest",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "temperature": 0.7
  }
]

analyzer_prompt = ( "1. Decompose the requirement into several easy-to-solve subproblems that"
    "can be more easily implemented by the developer."
    "2. Develop a high-level plan that outlines the major steps of the program."
    "Remember, your plan should be high-level and focused on guiding the developer in writing"
    "code, rather than providing implementation details."
    )

coder_prompt = (
    "Write code in Python that meets the requirements following the plan. Ensure"
    "that the code you write is efficient, readable, and follows best practices."
    "Remember, do not need to explain the code you wrote. Also, do not ask for more context."
)

repair_prompt = (
    "Fix or improve the code based on the content of the report. Ensure that any"
    "changes made to the code do not introduce new bugs or negatively impact the performance"
    "of the code. Remember, do not need to explain the code you wrote"
)

executor_prompt = (
    "1. Test the functionality of the code to ensure it satisfies the requirements."
    "2. Write reports on any issues or bugs you encounter."
    "3. If the code or the revised code has passed your tests, write a conclusion \"Code Test Passed\"."
    "Remember, the report should be as concise as possible, without sacrificing clarity and"
    "completeness of information. Do not include any error handling or exception handling"
    "suggestions in your report"
    "4. Output all code and prove that the code ran"
    "5. Do not modify the code in any way, your only job is to run the code and output the result"
    "Test the provided code and output results. Do not change the code in anyway."
)

test_writer_prompt = (
    "1. From the code provided to you, isolate the Python code snippet\n"
    "2. Write test cases within the Python code snippet that will run when tested, and comment their expected results based on the task, not the code.\n"
    "3. Output a single python code block, containing the original function and your test cases"
    "4. Only return the Python code snippet enclosed in code blocks (```) so your output can run on its own"
    "5. Never forget to fully enclose your response in backticks and specify that it is python code: ```python ... ```"
)

quality_assurance_prompt = (
    "Given the original task, the code and its output, determine if the code succeeded in achieving the task."
    "Do not alter the code in any way, return \"True\" if the code achieved the task and return \"False\" otherwise."
    "No other output is necessary"
)
   

#coder_prompt = AutoTransform(coder_prompt)
#repair_prompt = AutoTransform(repair_prompt)

analyzer_agent = autogen.AssistantAgent(
    name="analyzer",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message=analyzer_prompt,
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder_agent = autogen.AssistantAgent(
    name="coder",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message=coder_prompt,
    code_execution_config=False,
    human_input_mode="NEVER"
)

repair_agent = autogen.AssistantAgent(
    name="repair",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message=repair_prompt,
    code_execution_config=False,
    human_input_mode="NEVER"
)

tester_agent = autogen.AssistantAgent(
    name="tester",
    llm_config={"config_list": config_list, "temperature": 0.7},
    system_message=test_writer_prompt,
    human_input_mode="NEVER",
)

executor_agent = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list, "temperature": 0},
    human_input_mode="NEVER",
    system_message=executor_prompt,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)





analysis_result = analyzer_agent.generate_reply(
    messages=[{"content": Task, "role": "user"}]
)

coding_result = coder_agent.generate_reply(
    messages=[{"content": analysis_result, "role": "user"}]
)

repairing_result = repair_agent.generate_reply(
    messages=[{"content": coding_result, "role": "user"}]
)

test_result = tester_agent.generate_reply(
   messages=[{"content": repairing_result, "role": "user"}]
)
#doesn't always fully enclose code in ```, doesn't include python after ```, returns multiple blocks

executor_result = executor_agent.generate_code_execution_reply(
    messages=[{"content": test_result, "role": "user"}]
)




print("Analysis Result:", analysis_result)
print("Coding Result:", coding_result)
print("Repairing Result:", repairing_result)
print("Test Result:", test_result)
print("Code Running Result:", executor_result)

end = time.time()
print(end - start)