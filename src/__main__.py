import argparse
import json
from pathlib import Path
from typing import Any
from llm_sdk import Small_LLM_Model

def load_json_file(file_path: Path) -> Any:
    try:
        with file_path.open("r", endcoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {file_path}: {exc}") from exc


def save_json_file(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def build_prompt(functions: list[dict[str, Any]], user_prompt: str) -> str:
    return f"""
You are a function-calling system.

Available functions:
{json.dumps(functions, indent=2, ensure_ascii=False)}

User request:
{user_prompt}

Return ONLY a valid JSON object with exactly this format:
{{
    "name": "fcuntion_name"
    "parameters": {{
        "parameter1": "value"
    }}
}}

Rules:
- Choose exactly one function from the avialable functions.
- Use the exact function name.
- Use the exact parameter names from the function definition.
- Use correct parameter types.
- Return JSON only.
- Do not include markdown.
- Do not include explanations.
""".strip()