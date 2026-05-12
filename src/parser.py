from typing import Any, cast
from pydantic import ValidationError
from src.models import PromptInput, JsonValue, FunctionDefinition


def parse_tests(data: JsonValue) -> list[dict[str, str]]:
    """Validate and normalize test prompts."""
    if not isinstance(data, list):
        print("Error: test input must be a JSON list.")
        return []

    tests: list[dict[str, str]] = []

    for item in data:
        if not isinstance(item, dict):
            print("Warning: skipped invalid test item.")
            continue

        try:
            validated = PromptInput(**cast(dict[str, Any], item))  # **parse
        except ValidationError as error:
            print(f"Warning: skipped invalid test item: {error}")
            continue

        tests.append(validated.model_dump())  # trans pydantic object->dict

    return tests


def parse_functions(data: JsonValue) -> list[dict[str, Any]]:
    """Validate and normalize function definitions."""
    if not isinstance(data, list):
        print("Error: function definitions must be a JSON list.")
        return []

    functions: list[dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            print("Warning: skipped invalid function definition.")
            continue

        try:
            validated = FunctionDefinition(**cast(dict[str, Any], item))
        except ValidationError as error:
            print(f"Warning: skipped invalid function definition: {error}")
            continue

        functions.append(validated.model_dump())

    return functions
