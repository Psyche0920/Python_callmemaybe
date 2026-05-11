from typing import Any
from src.models import JsonValue


def parse_tests(data: JsonValue) -> list[dict[str, str]]:
    """Validate and normalize test prompts."""
    if not isinstance(data, list):
        return []
    tests: list[dict[str, str]] = []
    for item in data:
        if isinstance(item, dict):
            prompt = item.get("prompt")
            if isinstance(prompt, str):
                tests.append({"prompt": prompt})
    return tests


def parse_functions(data: JsonValue) -> list[dict[str, Any]]:
    """Validate and normalize function definitions."""
    if not isinstance(data, list):
        return []
    functions: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            functions.append(item)
    return functions
