from typing import Any
from pydantic import BaseModel, Field


class PromptInput(BaseModel):
    prompt: str


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


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
