import json
import math
from enum import Enum

import torch
from llm_sdk import Small_LLM_Model


class State(Enum):
    START = "start"
    OPEN_OBJECT = "open_object"
    NAME_KEY = "name_key"
    NAME_COLON = "name_colon"
    NAME_VALUE = "name_value"
    COMMA = "comma"
    PARAMETERS_KEY = "parameters_key"
    PARAMETERS_COLON = "parameters_colon"
    OPEN_PARAMETERS = "open_parameters"
    PARAM_NAME_KEY = "param_name_key"
    PARAM_NAME_COLON = "param_name_colon"
    PARAM_NAME_VALUE = "param_name_value"
    CLOSE_PARAMETERS = "close_parameters"
    CLOSE_OBJECT = "close_object"
    DONE = "done"


# 你只能使用 Enum 里“已经定义过的成员”，否则程序直接报错
# otherwise silent error and crash


def get_next_char(model, current_text, allowed_chars):
    """Choose the best next token restricted to allowed characters."""
    input_ids = model.encode(current_text)[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    filtered_logits = []

    for token_id, logit in enumerate(logits):
        token_text = model.decode([token_id])

        if token_text in allowed_chars:
            filtered_logits.append(logit)
        else:
            filtered_logits.append(-math.inf)
    
    next_token_id = torch.argmax(torch.tensor(filtered_logits)).item()
    next_char = model.decode([next_token_id])

    return next_char


def generate_fixed_text(current: str, text: str) -> str:
    """Applied fixed grammar text."""
    return current + text


def generate_limited_text(
        model: Small_LLM_Model, 
        current: str, 
        allowed_chars: set[str],
        length: int,
) -> str:
    """Generate a fixed number of constrained characters."""
    for _ in range(length):
        current += get_next_char(model, current, allowed_chars)
    return current


def generate_function_call_json() -> str:
    """Generate one function-call JSON object with an explicit state machine."""
    model = Small_LLM_Model()
    current = ""
    state = State.START

    function_chars = set("abcdefghijklmnopqrstuvwxyz_")
    name_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    while state != State.DONE:
        if state == State.START:
            current = generate_fixed_text(current, "{")
            state = State.NAME_KEY
        elif state == State.NAME_KEY:
            current = generate_fixed_text(current, '"name"')
            state = State.NAME_COLON
        elif state == State.NAME_COLON:
            current = generate_fixed_text(current, ': "')
            state = State.NAME_VALUE

        elif state == State.NAME_VALUE:
            current = generate_limited_text(
                model, current, function_chars, length=8,
            )
            current = generate_fixed_text(current, '"')
            state = State.COMMA

        elif state == State.COMMA:
            current = generate_fixed_text(current, ', ')
            state = State.PARAMETERS_KEY

        elif state == State.PARAMETERS_KEY:
            current = generate_fixed_text(current, '"parameters"')
            state = State.PARAMETERS_COLON

        elif state == State.PARAMETERS_COLON:
            current = generate_fixed_text(current, ": ")
            state = State.OPEN_PARAMETERS

        elif state == State.OPEN_PARAMETERS:
            current = generate_fixed_text(current, "{")
            state = State.PARAM_NAME_KEY

        elif state == State.PARAM_NAME_KEY:
            current = generate_fixed_text(current, '"name"')
            state = State.PARAM_NAME_COLON

        elif state == State.PARAM_NAME_COLON:
            current = generate_fixed_text(current, ': "')
            state = State.PARAM_NAME_VALUE

        elif state == State.PARAM_NAME_VALUE:
            current = generate_limited_text(
                model, current, name_chars, length=5
            )
            current = generate_fixed_text(current, '"')
            state = State.CLOSE_PARAMETERS

        elif state == State.CLOSE_PARAMETERS:
            current = generate_fixed_text(current, "}")
            state = State.CLOSE_OBJECT

        elif state == State.CLOSE_OBJECT:
            current = generate_fixed_text(current, "}")
            state = State.DONE

    return current


def main() -> None:
    """Run the JSON state machine demo."""
    result = generate_function_call_json()
    print(result)

    parsed = json.loads(result)
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()

# prompt
# → LLM 选择 name
# → 根据 name 找 schema
# → regex 提取 value
# → 按 schema 填 parameters
# → FunctionCall(BaseModel)
# → json.dump 输出

# 模型：选函数
# regex：提取参数
# schema：定义结构
# 代码：构造 JSON

# 下一步就是把你现在的 choose_function 升级成：模型选函数名，schema 决定参数名，regex 填参数值。这样就从 demo 变成真正项目逻辑。