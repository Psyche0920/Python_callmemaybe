import math

from llm_sdk import Small_LLM_Model
import torch

def main() -> None:
    """Test one step of constrained decoding."""

    model = Small_LLM_Model()

    current_text = "{"

    # 1️⃣ encode（文本 → token）
    input_ids = model.encode(current_text)[0].tolist()


    #  2️⃣ 模型（token → logits）
    logits = model.get_logits_from_input_ids(input_ids)

    # token 0: "a"     score = 1.2
    # token 1: "b"     score = 0.5
    # token 2: "\""    score = 8.9
    # token 3: "}"     score = 2.1
    # token 4: "\n"    score = 3.0

    # 所以模型认为下一个最可能是双引号。


    # 3️⃣ 选最大概率 token
    next_token_id = torch.argmax(torch.tensor(logits)).item()

    # torch.tensor(logits) -> tensor

    # torch.argmax(tensor)-> tensor(index) 
    # e.g. logits = [1.2, 0.5, 8.9, 2.1, 3.0]
    # return tensor(2)

    # .item() 是 PyTorch tensor 的一个方法。
    # .item() = 把“只有一个值的 tensor”变成 Python 的普通数值
    # tensor(2)= 2

    # 4️⃣ decode（token → 文本）
    next_token_text = model.decode([next_token_id])


    print("current:", current_text)
    print("next_token_id:", next_token_id)
    print("decoded:", repr(next_token_text))
# repr() = 把内容用“程序员能看清的形式”显示出来。会显示：'\n'

# text->(token ids)->list->tensor-> logits(list[float])->argmax(tensor)->int->tensor or list[int]-> text

# # 1. tensor format（数据结构）
# 0D: tensor(2)         → 标量（一个值）
# 1D: tensor([2,5,8])   → 一个序列 / 向量 一维 一层 []：表示一个 token 序列（sequence）
# 2D: tensor([[2,5,8]]) → batch + sequence  二维 两层 []：

# # 2. token id sequence format（数据语义）
# 单个序列：
# [2,5,8]

# batch（多个序列）：
# [
#   [2,5,8],
#   [9,10]



# 2. tensor vs tokenizer

##模型（语言模型）
# 模型内部使用 tensor 进行计算（GPU / 向量化）
# 输入和输出在底层都是 tensor

# tensor 是一个通用数值容器（多维数组）
# tensor 本身不包含语义，
# 其含义取决于数据（可以表示 token ids / embeddings / logits / probabilities）

# tokenizer（编码解码） 负责处理 text 和 token id 序列
# encode: text → sequence of token ids（list[int] 或 tensor）
# decode: sequence of token ids → text

# decode 通常使用 list[int]（简单、安全，不涉及计算）
# 在工程中：
# encode 有时直接返回 tensor（由 SDK 设计，方便喂模型）
# decode 使用 list[int] 更常见（高层接口更简单）


# 3. token ≠ 字符
# token 可以是：
# 类型	示例
# 字符	{
# 单词	hello
# 子词	ing, un
# 空格	" "
# 标点	,

# [Summarize]
# 1. tokenizer：负责 text ↔ token ids（整数序列）

# 2. 模型（内部用 tensor 计算）：
#    token ids → embedding（向量） → Transformer（理解上下文） → logits（每个 token 的分数）

# 3. 生成：
#    从 logits 选择下一个 token id（argmax / sampling），
#    再通过 tokenizer.decode 转回文本
