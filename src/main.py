import argparse
import json
from typing import Any

from llm_sdk import Small_LLM_Model

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
    # argparse 是一个模块（库）
    # ArgumentParser 是这个模块里的一个类（class）, ArgumentParser = 参数解析器（专门处理终端输入）
    # add_argument = 定义可以输入什么
    # parse_args   = 读取用户输入
    # args.xxx     = 使用这个值

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

# 读取命令行参数

# --functions_definition
# --input
# --output
# 读取 JSON 文件
# functions_definition.json：可调用函数 schema
# function_calling_tests.json：prompt 列表
# 用 LLM 选择函数
# 对每个 prompt，把所有 function name 当候选
# 用 Small_LLM_Model
# 用 get_logits_from_input_ids() 给每个 function name 打分
# 选分数最高的函数
# 根据选中的函数提取参数
# add：提取两个数字
# square_root：提取一个数字
# greet：提取 greet 后面的名字
# reverse：提取引号里的字符串
# substitute/regex：提取 source_string / regex / replacement

# 写入 output JSON
# 每条结果结构是：

# {
#   "prompt": "...",
#   "name": "...",
#   "args": {}
# }
