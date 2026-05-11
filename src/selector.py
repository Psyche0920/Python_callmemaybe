import math
from typing import Any
from llm_sdk import Small_LLM_Model


def softmax(scores: list[float]) -> list[float]:
    """Convert scores to probabilities"""
    exp_scores = [math.exp(score) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


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

    names: list[str] = []
    scores: list[float] = []

    for function in functions:
        name = str(function.get("name", ""))
        suffix = name[2:] if name.startswith("fn") else name
        suffix_ids = model.encode(suffix)[0].tolist()

        if not suffix_ids:
            continue

        score = float(logits[suffix_ids[0]])

        names.append(name)
        scores.append(score)

    if not scores:
        print(f"Warning: no valid function candidate for prompt: {prompt!r}")
        return ""

    probabilities = softmax(scores)

    best_index = max(range(len(probabilities)),
                     key=lambda index: probabilities[index])

    best_probability = probabilities[best_index]

    if best_probability < 0.55:
        print(f"Warning: ambiguous/no function match for prompt: {prompt!r}")
        return ""

    return names[best_index]

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

# selector = 在合法函数里选最像的--softmax
# decoder = 在所有 token 里禁止非法 token，强制生成合法 JSON---mathinf

# 所以方式不同是正常的。selector 是 selection/scoring，decoder 是 generation constraint。
