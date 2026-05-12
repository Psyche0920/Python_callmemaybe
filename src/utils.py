import json
from pathlib import Path
from typing import cast

from src.models import FunctionCall, JsonValue


def load_json(path: str) -> JsonValue | None:
    """Read a JSON file and return its content."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return cast(JsonValue, data)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
    except PermissionError:
        print(f"Error: permission denied: {path}")
    except UnicodeDecodeError:
        print(f"Error: file is not valid UTF-8 {path}")
    except json.JSONDecodeError:
        print(f"Error: invalid JSON file: {path}")
    return None


def write_json(path: str, data: list[FunctionCall]) -> None:
    """Write JSON output to a file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump([item.model_dump() for item in data], file, indent=2)
    print(f"Output written to {output_path}")
