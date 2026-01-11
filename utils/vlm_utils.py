import numpy as np
import os
import random
import base64
from openai import OpenAI, AsyncOpenAI

random.seed(123)  # Set random seed to 123

async def Qwen_Async(image_path, prompt):
    # Base64 encoding of the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)
    client = AsyncOpenAI(
        # If the environment variable is not configured, replace the line below with your API Key: api_key="sk-xxx"
        api_key='sk-6ef267af2c014abbbb8b1c9eabbc127b',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = await client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # Using f-string to create a string containing the BASE64 encoded image data.
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

async def GPT4V_Async(image_path, prompt):
    client = AsyncOpenAI(api_key="your_api_key")

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0]

def Qwen(image_path, prompt):
    # Base64 encoding of the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)
    client = OpenAI(
        # If the environment variable is not configured, replace the line below with your API Key: api_key="sk-xxx"
        api_key='sk-6ef267af2c014abbbb8b1c9eabbc127b',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # Using f-string to create a string containing the BASE64 encoded image data.
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def GPT4V(image_path, prompt):
    client = OpenAI(api_key="your_api_key")

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0]


def get_image_files(directory):
    image_files = []

    # Original code only processed folder "01":
    # for i in range(1, 2):
    #     sub_path = str(i).zfill(2)
    #     ...
    #
    # New behavior: iterate all subfolders under gpt_input (each is a view like 01/02/...).
    if not os.path.exists(directory):
        return []
    subfolders = [
        d
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]
    # Prefer numeric 2-digit folders, but keep a fallback for any folder names.
    def _key(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    for sub in sorted(subfolders, key=_key):
        now_path = os.path.join(directory, sub)
        for png in os.listdir(now_path):
            if not png.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(now_path, png)
            image_files.append(img_path)

    image_files = sorted(image_files, key=lambda x: (os.path.basename(os.path.dirname(x)), os.path.basename(x)))

    return image_files




