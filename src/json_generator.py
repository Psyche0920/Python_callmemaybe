import math
import torch
from llm_sdk import Small_LLM_Model

def get_next_char(model, current_text, allowed_chars):
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

def generate_json():
    model = Small_LLM_Model()
    current = ""

    # state machine
    steps = [
        # {
        {"type": "fixed", "value": "{"},
        
        # "name"
        {"type": "fixed", "value": '"name": "' },
        {"type": "generate", "allowed": "abcdefghijklmnopqrstuvwxyz_"},
        {"type": "generate", "allowed": "abcdefghijklmnopqrstuvwxyz_"},
        {"type": "generate", "allowed": "abcdefghijklmnopqrstuvwxyz_"},
        {"type": "fixed", "value": '",'},

        # parameters
        {"type": "fixed", "value": '"parameters": {'},

        # parameters.name
        {"type": "fixed", "value": '"name": "'},
        {"type": "generate", "allowed": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"},
        {"type": "generate", "allowed": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"},
        {"type": "generate", "allowed": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"},
        {"type": "fixed", "value": '"'},
        
        # close parameters
        {"type": "fixed", "value": "}"},

        # close object
        {"type": "fixed", "value": "}"},
    ]

    for step in steps:
        if step["type"] == "fixed":
            current += step["value"]
        elif step["type"] == "generate":
            allowed_chars = set(step["allowed"])
            next_char = get_next_char(model, current, allowed_chars)
            current += next_char
        print(current)
    return current


if __name__ == "__main__":
    result = generate_json()
    print("\nFINAL JSON:")
    print(result)





# 👉 这段代码的核心思想是：

# ❗ 不用模型自由生成，而是你控制 JSON 结构，只让模型填“字符”

# 也就是：

# 结构（{ "name": ... }）👉 你写死
# 内容（name、参数）👉 模型选
# 但模型也不能乱选 👉 只能从 allowed_chars 里面选

# 👉 这就是：

# 💡 constrained decoding（受限解码）
