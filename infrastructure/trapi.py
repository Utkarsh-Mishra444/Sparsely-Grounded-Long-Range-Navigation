import os
import requests
import base64
import mimetypes
from pathlib import Path
from typing import Union
import json

ENDPOINT = os.environ.get("LLM_PROXY_URL", "") + "/chat/completions" if os.environ.get("LLM_PROXY_URL") else ""
PROXY_API_KEY = os.environ.get("LLM_PROXY_KEY", "")

def _image_to_data_url(path: Union[str, Path]) -> str:
    p = Path(path)
    mime_type, _ = mimetypes.guess_type(p.name)
    mime_type = mime_type or "image/jpeg"

    with p.open("rb") as f:
        b64_data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{b64_data}"


def ask_trapi_with_image(question: str, image_path: str, *, model: str = None, temperature: float = 0.2) -> str:
    data_url = _image_to_data_url(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "stream": False,
        # "temperature": temperature,
    }
    headers = {
        "X-API-Key": PROXY_API_KEY,
        "x-functions-key": PROXY_API_KEY,
        "Content-Type": "application/json",
    }
    resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        print(f"Proxy error: {resp.status_code} {resp.text[:1000]}")
        resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


if __name__ == "__main__":
    image_path = "/Users/dwipdalal/Desktop/Spatail Reasoning Check/image1.jpg"
    prompt = "keep the answers in one line "
    # question = "What would happen when I push the first chair near the camera toward right side of the chair?"
    question = "This is the camera feed of a legged quadraped navigating to the black door. Should the robot go right or left of the rubble and in short answer why?"
    # list of names: gpt-5, gpt-4o, gpt-5-chat, o3, gpt-4.1, o1, gpt-5-mini and gpt-5-nano
    model = 'gpt-5'
    try:
        answer = ask_trapi_with_image(prompt + question, image_path, model=model)
    except Exception as e:
        raise
    print("\nAssistant:", answer)
    # Save the result in JSON format
    asked_question = prompt + question
    record = {
        "image": Path(image_path).name,
        "Question": asked_question,
        "Answer": answer,
        "model": model,
    }

    output_json_path = Path('/Users/dwipdalal/Desktop/Spatail Reasoning Check/output.json')

    if output_json_path.exists():
        try:
            with output_json_path.open('r', encoding='utf-8') as f:
                data_list = json.load(f)
                if not isinstance(data_list, list):
                    data_list = []
        except json.JSONDecodeError:
            data_list = []
    else:
        data_list = []

    data_list.append(record)

    with output_json_path.open('w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2)
    