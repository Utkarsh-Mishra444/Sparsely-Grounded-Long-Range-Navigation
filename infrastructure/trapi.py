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
    if not ENDPOINT:
        raise RuntimeError("LLM_PROXY_URL is not set; cannot call TRAPI proxy.")
    if not PROXY_API_KEY:
        raise RuntimeError("LLM_PROXY_KEY is not set; cannot call TRAPI proxy.")
    resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        print(f"Proxy error: {resp.status_code} {resp.text[:1000]}")
        resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Quick TRAPI-proxy image+text query helper.")
    p.add_argument("--image", required=True, help="Path to an input image file")
    p.add_argument("--model", default="gpt-5", help="Upstream model name for the proxy (e.g., gpt-5, gpt-4o)")
    p.add_argument("--prompt", default="", help="Optional prompt prefix")
    p.add_argument("--question", required=True, help="Question to ask about the image")
    p.add_argument("--out", default="trapi_output.json", help="Append results to this JSON file (set empty to disable)")
    args = p.parse_args()

    answer = ask_trapi_with_image(args.prompt + args.question, args.image, model=args.model)
    print("\nAssistant:", answer)

    if args.out:
        output_json_path = Path(args.out)
        record = {
            "image": Path(args.image).name,
            "Question": args.prompt + args.question,
            "Answer": answer,
            "model": args.model,
        }

        if output_json_path.exists():
            try:
                with output_json_path.open("r", encoding="utf-8") as f:
                    data_list = json.load(f)
                if not isinstance(data_list, list):
                    data_list = []
            except json.JSONDecodeError:
                data_list = []
        else:
            data_list = []

        data_list.append(record)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2)
    
