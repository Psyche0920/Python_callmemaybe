import argparse
import json
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.decoder import SchemaConstrainedDecoder
from src.extracter import build_args
from src.models import FunctionCall
from src.parser import parse_functions, parse_tests
from src.selector import choose_function_name_with_llm
from src.utils import load_json, write_json


def find_function_schema(
    function_name: str,
    functions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the function schema by function name.// return the dict object."""
    for function in functions:
        if function.get("name") == function_name:
            return function
    return None


def make_function_call(
    model: Small_LLM_Model,
    decoder: SchemaConstrainedDecoder,
    prompt: str,
    functions: list[dict[str, Any]],
) -> FunctionCall:
    """Create one schema-valid function-call result."""
    function_name = choose_function_name_with_llm(model, prompt, functions)
    function_schema = find_function_schema(function_name, functions)

    if function_schema is None:
        parameters: dict[str, Any] = {}
    else:
        parameters = build_args(prompt, function_schema)

    json_text = decoder.generate_function_call_json(
        prompt=prompt,
        function_name=function_name,
        function_schema=function_schema,
        parameters=parameters
    )
    parsed = json.loads(json_text)
    return FunctionCall(**parsed)


def main() -> None:
    """Run the function-calling pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )

    args = parser.parse_args()
    # ArgumentParser arguments parsing module
    # add_argument = add arguments
    # parse_args   = read arguments
    # args.xxx     = use

    raw_functions = load_json(args.functions_definition)
    raw_tests = load_json(args.input)

    if raw_functions is None or raw_tests is None:
        return

    functions = parse_functions(raw_functions)
    tests = parse_tests(raw_tests)

    model = Small_LLM_Model()
    decoder = SchemaConstrainedDecoder(model)

    results: list[FunctionCall] = []

    for test in tests:
        prompt = test["prompt"]
        try:
            result = make_function_call(model, decoder, prompt, functions)
        except Exception as error:
            print(f"Error: could not process prompt {prompt!r}: {error}")
            result = FunctionCall(prompt=prompt, name="", parameters={})
        results.append(result)

    write_json(str(args.output), results)

# Enum 用来表示有限固定状态；
# BaseModel 用来表示结构化数据对象
