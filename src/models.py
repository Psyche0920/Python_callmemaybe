from typing import Any
from pydantic import BaseModel


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


JsonValue = (
    dict[str, "JsonValue"]
    | list["JsonValue"]
    | str
    | int
    | float
    | bool
    | None
)
