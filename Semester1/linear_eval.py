import autogen
from auto_transform import *
import random

def run_mas(Task):

    config_list = [
        {
            "model": "gpt-3.5-turbo",
            "api_key": openai_api_key,
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

    code_output_prompt = (
        "Given the python code snippet, output only the code. Do not include any (```) around the code. Output only the code within them"
    )

    analyzer_agent = autogen.AssistantAgent(
        name="analyzer",
        llm_config={"config_list": config_list},
        system_message=analyzer_prompt,
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    coder_agent = autogen.AssistantAgent(
        name="coder",
        llm_config={"config_list": config_list},
        system_message=coder_prompt,
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    repair_agent = autogen.AssistantAgent(
        name="repair",
        llm_config={"config_list": config_list},
        system_message=repair_prompt,
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    code_output_agent = autogen.AssistantAgent(
        "outputer",
        llm_config={"config_list": config_list},
        system_message=code_output_prompt,
        human_input_mode="NEVER"
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

    code_output = code_output_agent.generate_reply(
        messages=[{"content": repairing_result, "role": "user"}]
    )

    print(code_output)

run_mas(Task_2)
