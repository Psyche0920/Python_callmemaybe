import argparse
import json
import re
from pathlib import Path
from typing import Any, cast
from pydantic import BaseModel
from llm_sdk import Small_LLM_Model


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


def extract_numbers(prompt: str) -> list[float]:
    """Extract numbers from a prompt."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    return [float(match) for match in matches]


def extract_quoted_strings(prompt: str) -> list[str]:
    """Extract all texts inside single or double quotes."""
    return re.findall(r"['\"]([^'\"]*)['\"]", prompt)


def extract_first_quoted_string(prompt: str) -> str:
    """Extract the first quoted string."""
    quoted = extract_quoted_strings(prompt)
    return quoted[0] if quoted else ""
    # ???


def extract_greet_name(prompt: str) -> str:
    """Extract the name after the word greet."""
    match = re.search(
        r"greet\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # * >=0, + >= 1
        prompt,
        re.IGNORECASE,
    )
    return cast(str, match.group(1)) if match else ""
    # greet\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)
    #         ↑ 这个括号是 group(1)
    # match.group()     → "Greet Alice Smith"
    # match.group(1)    → "Alice Smith"


def score_candidate_name(
        model: Small_LLM_Model,
        context: str,
        candidate_name: str,
) -> float:
    """Score a function name token by token under a fixed context."""
    current = context
    total = 0.0
    candidate_ids = model.encode(candidate_name)[0].tolist()
    
    for token_id in candidate_ids:
        input_ids = model.encode(current)[0].tolist()
        logits = model.get_logits_from_input_ids(input_ids)
        total += logits[token_id]
        current += model.decode([token_id])
    return total

# 🎯 关键总结（帮你锁死）

# 👉 vocab size：

# ✔ 固定 → 决定 logits 长度

# 👉 logits：

# ✔ 每次根据 context 重新计算
# ✔ 给“所有 token”打分

# 👉 你的代码：

# ✔ 一步步验证 candidate_name 是否“符合模型习惯”


def choose_function_name_with_llm(
        model: Small_LLM_Model,
        prompt: str, 
        functions: list[dict[str, Any]],
) -> str:
    """Choose the best function name using LLM logits."""
    if not functions:
        return ""
    options = "\n".join(
        f"- {function.get('name')}: {function.get('description', '')}"
        for function in functions
    )
    context = (
        "Choose the best function for the user request.\n"
        f"Available functions:\n{options}\n"
        f"User request: {prompt}\n"
        "Best function name: "
    )
    best_name = ""
    best_score = float("-inf")

    for function in functions:
        name = str(function.get("name", ""))
        score = score_candidate_name(model, context, name)

        if score > best_score:
            best_score = score
            best_name = name
    return best_name

def make_empty_result(prompt: str) -> FunctionCall:
    """Create an empty fallback result."""
    return FunctionCall(
        prompt=prompt,
        name="",
        parameters={},
    )


def score_text(model: Small_LLM_Model, text: str) -> float:
    """Score how likely a text is under the model."""
    ids = model.encode(text)[0].tolist()
    total = 0.0

    for index in range(1, len(ids)):
        prefix = ids[:index]
        target = ids[index]
        logits = model.get_logits_from_input_ids(prefix)
        total += logits[target]
    return total


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