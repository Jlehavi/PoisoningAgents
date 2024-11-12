import random
import openai
from openai import OpenAI


api_key = "----------------------------------------------------" # Replace with your OpenAI API key
client = OpenAI(api_key=api_key)

def AutoInject(message, MessageModifyRate, SentenceModifyRate, model, TaskDescription, ModifyDescription, api_key):

    # Decide whether to modify the message
    if random.random() > MessageModifyRate:
        # Do not modify the message
        return message

    # Split the message into lines for processing
    lines = message.split('\n')

    modified_lines = []
    for line in lines:
        if random.random() <= SentenceModifyRate and line.strip() != '':
            # Modify this line using the AI model
            prompt = f"""Task Description: {TaskDescription}

Modify the following line according to these instructions:
{ModifyDescription}

Original Line:
{line}

Modified Line (only output the modified line, do not include any explanations):"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150,
                n=1,
                stop=None,
            )
            modified_line = response.choices[0].message.content
            modified_lines.append(modified_line)
        else:
            # Keep the original line
            modified_lines.append(line)

    # Reconstruct the modified message
    modified_message = '\n'.join(modified_lines)
    return modified_message
