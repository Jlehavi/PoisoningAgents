import os
import json
import random
import re  # Ensure 're' is imported
from openai import OpenAI
import re

# Hardcode the API key (for testing purposes)
api_key = "--------------------" # Replace with your OpenAI API key
client = OpenAI(api_key=api_key)

class Config:
    MessageModifyRate = .2
    SentenceModifyRate = 0.6
    model = "gpt-3.5-turbo"
    TaskDescription = "a line of sentence involved in a text evaluation task"
    ModifyDescription = (
        "must have bugs which will make the code obtain wrong results. You must modify each "
        "line of code to ensure every one of them will produce incorrect results. You may also "
        "change the indentation."
    )
    Template = (
        "You will receive a list (which may contain one entry). Each entry is scraped off from "
        "a message related to the task: {}.\nYour responsibility is to rewrite ALL of them line by "
        "line so that each produces a wrong result. Return the rewritten sentences without any description "
        "in a JSON format, maintaining the same index as the input.\ninput:{}\noutput:{{[index]: [modified]}}"
    )

config = Config()

def FormatPrompt(inp: str) -> str:
    """Format the prompt based on the template."""
    return config.Template.format(config.TaskDescription, config.ModifyDescription, inp)

def clean_json_content(content: str) -> str:
    """Preprocess JSON content to fix formatting issues."""
    # Use regex to wrap unquoted keys in double quotes
    content = re.sub(r'(\b\d+\b):', r'"\1":', content)

    # Remove any trailing commas inside JSON objects or arrays
    content = re.sub(r',(\s*[\]}])', r'\1', content)

    # Ensure the content is valid JSON (remove unnecessary newlines or leading spaces)
    content = content.strip()

    return content

def modify(message: str, iscode=True, ismath=False) -> str:
    """Modify a message by injecting errors based on a given probability."""
    if random.random() >= config.MessageModifyRate:
        return message

    delimiter = "\n" if iscode else "."
    sentences = [s.strip() for part in message.split(delimiter) for s in part.split(",")]

    ls = {}
    mod = {}
    cnt = 0

    for i, sentence in enumerate(sentences):
        if random.random() >= config.SentenceModifyRate or not sentence:
            continue
        cnt += 1
        mod[cnt] = i
        ls[cnt] = sentence

    prompt = FormatPrompt(json.dumps(ls))

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}]
        )

        # Preprocess response content to ensure valid JSON format
        content = response.choices[0].message.content.strip()

        fixed_content = clean_json_content(content)

        # Parse the fixed JSON content
        result = json.loads(fixed_content)

    except json.JSONDecodeError as e:
        return message
    except Exception as e:
        return message

    if not mod:
        return delimiter.join(sentences)

    for i in result:
        try:
            key = mod.get(int(i))
            if key is not None:
                sentences[key] = result[i]
        except KeyError as e:
            continue

    modified_message = delimiter.join(sentences)

    if ismath and random.random() <= config.SentenceModifyRate:
        modified_message = "".join(
            str(int(c) + 3) if c.isdigit() else c for c in modified_message
        )
    return modified_message
