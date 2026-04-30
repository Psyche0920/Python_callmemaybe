import argparse
import json
import re
from pathlib import Path
from typing import Any, cast
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

# 👉 JSON：
# 没有显式类型定义（implicit typing）
# 👉 Python：
# 需要显式类型（type hints）来检查代码
# 👉 所以你写的：
# JsonValue = ...
# 其实是在：
# ✅ 用 Python 模拟 JSON 的类型系统
# ✅ 让 mypy 能检查
# ✅ 满足 42 项目要求
# "JsonValue" = 防止类型还没定义时报错

# ✔ null（JSON） = 数据的一部分（合法值）
# ✔ None（Python） =
#     ① 数据里的空值（对应 null）
#     ② 程序员用来表示“没有返回 / 出错”（一种约定）


def load_json(path: str) -> JsonValue | None:
    """Read a JSON file and return its content."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return cast(JsonValue, data)
        # cast = 告诉类型检查器（mypy）：“相信我，这个值就是这个类型”
        # 函数	方向	用途
        # load	文件 → Python	读取
        # dump	Python → 文件	写入
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
    except PermissionError:
        print(f"Error: permission denied: {path}")
    except UnicodeDecodeError:
        print(f"Error: file is not valid UTF-8 {path}")
    except json.JSONDecodeError:
        print(f"Error: invalid JSON file: {path}")
    return None

# FileNotFoundError
# = 文件都没找到
# 先保证文件存在

# UnicodeDecodeError
# = 文件找到了，但 bytes 不能按 UTF-8 变成文字
# encoding="utf-8"：
# 告诉 Python：用全球标准方式解释文本，避免乱码和报错
# 再保证能用 UTF-8 读

# JSONDecodeError
# = 文字读出来了，但不是合法 JSON
# JSONDecodeError: Python 在解析 JSON 时失败（decode 失败）
# 本质原因（一句话）: 👉 文件内容不是“合法 JSON”
# 再保证是 valid JSON

# ValidationError
# = JSON 合法，但内容不符合你项目要求
# 再保证符合 function_definitions / output schema


def extract_numbers(prompt: str) -> list[float]:
    """Extract numbers from a prompt."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    return [float(match) for match in matches]


def extract_quoted_string(prompt: str) -> str:
    """Extract text inside single quotes."""
    match = re.search(r"'([^']*)'", prompt)
    if match is None:
        return ""
    value = match.group(1)
    if isinstance(value, str):
        return value
    return ""


def extract_greet_name(prompt: str) -> str:
    """Extract the name after the word greet."""
    match = re.search(
        r"greet\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        prompt,
        re.IGNORECASE,
    )
    if match is None:
        return ""
    return cast(str, match.group(1))
    # greet\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)
    #         ↑ 这个括号是 group(1)
    # match.group()     → "Greet Alice Smith"
    # match.group(1)    → "Alice Smith"


def make_empty_result(prompt: str) -> FunctionCall:
    """Create an empty fallback result."""
    return FunctionCall(
        prompt=prompt,
        name="",
        parameters={},
    )


def choose_function(
    prompt: str,
    functions: list[dict[str, Any]],
) -> FunctionCall:
    """Choose a function and extract parameters using simple rules."""
    lower_prompt = prompt.lower()
    for function in functions:
        name = str(function.get("name", ""))
        if ("add" in name or "sum" in name) and (
            "sum" in lower_prompt or "add" in lower_prompt
        ):
            numbers = extract_numbers(prompt)
            return FunctionCall(
                prompt=prompt,
                name=name,
                parameters={
                    "a": numbers[0] if len(numbers) > 0 else 0.0,
                    "b": numbers[1] if len(numbers) > 1 else 0.0,
                },
            )

        if "greet" in name and "greet" in lower_prompt:
            person_name = extract_greet_name(prompt)
            # \d+: 一个或多个数字
            # \s+         → 一个或多个空格
            # [] = 字符集合（character class）
            # (...)*: 这个“组”可以重复 0 次或多次
            # (...)	分组
            # (?:...) 分组但不捕获
            return FunctionCall(
                prompt=prompt,
                name=name,
                parameters={"name": person_name},
            )

        if "reverse" in name and "reverse" in lower_prompt:
            return FunctionCall(
                prompt=prompt,
                name=name,
                parameters={
                    "s": extract_quoted_string(prompt),
                },
            )
    return make_empty_result(prompt)


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


# parse_tests → 只需要 prompt → 做“数据清洗”
# parse_functions → 需要完整结构 → 直接保留
# schema 必须完整（subject 要求）
# 函数	处理方式	原因
# parse_tests	重建 dict	只需要 prompt + 清洗
# parse_functions	保留 item	结构复杂 + 必须完整
# tests = data（可以过滤、简化）
# functions = schema（必须完整、不能改）


def write_json(path: str, data: list[FunctionCall]) -> None:
    """Write JSON output to a file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump([item.model_dump() for item in data], file, indent=2)
    print(f"Output written to {output_path}")

    # encoding = UTF-8 是规则
    # read  → 用这个规则 decode
    # write → 用这个规则 encode

    # json.dump(data, file, indent=2)
    # Python对象 → JSON字符串 → 写入文件
    # data (Python list/dict)
    # ↓
    # json.dump
    # ↓
    # JSON字符串
    # ↓
    # file.write
    # ↓
    # 磁盘文件（.json）


def main() -> None:
    """Run the first simple version of the project."""
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
    functions = parse_functions(raw_functions)
    tests = parse_tests(raw_tests)

    results: list[FunctionCall] = []

    for test in tests:
        prompt = test["prompt"]
        result = choose_function(prompt, functions)
        results.append(result)

    write_json(str(args.output), results)


if __name__ == "__main__":
    main()

# v2:

# ✔ JSON 读取（load_json）
# ✔ 数据清洗（parse_tests / parse_functions）
# ✔ rule-based function selection (（if + regex）)=> subject 要求：LLM + constrained decoding
# ✔ regex 参数提取
# ✔ 输出 JSON
# ✔ argparse CLI
# ✔ pydantic 结构
# ✔ makefile

# now: ✔ logits → token → decode
# "{"
#    ↓ encode
# token ids
#    ↓ model
# logits（所有可能字符）
#    ↓ argmax
# 选一个 token
#    ↓ decode
# 变成字符