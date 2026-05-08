import argparse
import json
import re
import math
from pathlib import Path
from typing import Any, cast
from pydantic import BaseModel
from llm_sdk import Small_LLM_Model


class FunctionCall(BaseModel):
    prompt: str
    fn_name: str
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
    matches = re.findall(r"(['\"])(.*?)\1", prompt)
    return [match[1] for match in matches]
# “匹配任意内容，但尽可能少匹配”也叫：非贪婪匹配（non-greedy / lazy match）
# ( )capture group
# .任意字符
# *重复 0 次或无限次
# ? -> non-greedy 能少吃就少吃

def extract_first_quoted_string(prompt: str) -> str:
    """Extract the first quoted string."""
    quoted = extract_quoted_strings(prompt)
    return quoted[0] if quoted else ""


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


def choose_function_name_with_llm(
        model: Small_LLM_Model,
        prompt: str, 
        functions: list[dict[str, Any]],
) -> str:
    """Choose function using LLM logits at the first token after 'fn'."""
    if not functions:
        return ""
    
    options = "\n".join(
        f"- {function.get('name')}: {function.get('description', '')}"
        for function in functions
    )

    context = (
        "Choose exactly one function from the list.\n"
        f"Available functions:\n{options}\n"
        f"User request: {prompt}\n"
        "The correct function is: fn"
    )

    input_ids = model.encode(context)[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    best_name = ""
    best_score = -math.inf

    for function in functions:
        name = str(function.get("name", ""))
        suffix = name[2:] if name.startswith("fn") else name
        suffix_ids = model.encode(suffix)[0].tolist()
        
        if not suffix_ids:
            continue
        
        score = logits[suffix_ids[0]]

        if score > best_score:
            best_score = score
            best_name = name
    return best_name

# encode(context)
# ↓
# 得到 context token ids

# get_logits_from_input_ids(input_ids)
# ↓
# 得到“下一个 token”的 logits 分布

# encode(suffix)
# ↓
# 得到 function suffix 对应 token ids

# logits[suffix_ids[0]]
# ↓
# 查看模型对：
# “下一个 token 是这个 suffix 第一个 token”
# 的偏好分数


def find_function_schema(
    function_name: str,
    functions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the function schema by function name.// return the dict object."""
    for function in functions:
        if function.get("name") == function_name:
            return function
    return None


def get_parameter_names(function_schema: dict[str, Any]) -> list[str]:
    """Return parameter names from the function schema."""
    parameters = function_schema.get("parameters", {})
    # parameters = {"a": "int", "b": "int"}
    if not isinstance(parameters, dict):
        return []
    return list(parameters.keys())
    # ["a", "b"]


def build_substitute_args(
    prompt: str,
    parameter_names: list[str],
) -> dict[str, Any]:
    """Build args for regex substitution."""
    quoted = extract_quoted_strings(prompt)
    lower = prompt.lower()
    source_string = quoted[-1] if quoted else ""

    if "numbers" in lower:
        regex = r"\d+"
    elif "vowels" in lower:
        regex = r"[aeiouAEIOU]"
    elif quoted:
        regex = re.escape(quoted[0])
        # re.escape() 是为了防止特殊字符出问题，比如 .、*、?。
        # 如果没有 numbers / vowels，但有引号，就把第一个引号里的内容当作要匹配的东西。
    else:
        regex = ""

    if "asterisks" in lower:
        replacement = "*"
    else:
        match = re.search(r"with\s+['\"]?([A-Za-z0-9_*]+)['\"]?", prompt)
        replacement = match.group(1) if match else ""
    values = {
        "source_string": source_string,
        "regex": regex,
        "replacement": replacement,
    }
    return {name: values.get(name, "") for name in parameter_names}


def build_args(
    prompt: str,
    function_schema: dict[str, Any],
) -> dict[str, Any]:
    """Build function arguments according to the selected schema."""
    function_name = str(function_schema.get("name", "")).lower()
    parameter_names = get_parameter_names(function_schema)
    # ["a", "b"]

    if not parameter_names:
        return {}
    
    if "add" in function_name or "sum" in function_name:
        numbers = extract_numbers(prompt)
        return {
            name: numbers[index] if index < len(numbers) else 0.0
            for index, name in enumerate(parameter_names)
        }
        # {"a": 3, "b": 5}
    if "square" in function_name or "sqrt" in function_name:
        numbers = extract_numbers(prompt)
        return {
            parameter_names[0]: numbers[0] if numbers else 0.0
        }
        # {"a": 144}
    if "greet" in function_name:
        return {
            parameter_names[0]: extract_greet_name(prompt)
        }
    # {"a": "Ann"}
    if "reverse" in function_name:
        return {
            parameter_names[0]: extract_first_quoted_string(prompt)
        }
    if "substitute" in function_name or "replace" in function_name:
        return build_substitute_args(prompt, parameter_names)
    return {name: "" for name in parameter_names}


def make_function_call(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[dict[str, Any]],
) -> FunctionCall:
    """Create one valid function-call result."""
    function_name = choose_function_name_with_llm(model, prompt, functions)
    function_schema = find_function_schema(function_name, functions)

    if function_schema is None:
        return FunctionCall(prompt=prompt, fn_name="", parameters={})
    
    return FunctionCall(
        prompt=prompt,
        fn_name=function_name,
        parameters=build_args(prompt, function_schema),
    )


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

    results: list[FunctionCall] = []

    for test in tests:
        prompt = test["prompt"]
        result = make_function_call(model, prompt, functions)
        results.append(result)

    write_json(str(args.output), results)


if __name__ == "__main__":
    main()


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
#   "fn_name": "...",
#   "args": {}
# }