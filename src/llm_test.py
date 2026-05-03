import math

from llm_sdk import Small_LLM_Model
import torch

def main() -> None:
    """Test one step of constrained decoding."""

    model = Small_LLM_Model()

    current_text = "{"

    input_ids = model.encode(current_text)[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    allowed_chars = set(
        '{}[]:,"abcdefghijklmnopqrstuvwxyz'
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    )

    filtered_logits: list[float] = []
    
    for token_id, logit in enumerate(logits):
        token_text = model.decode([token_id])
        if token_text in allowed_chars:
            filtered_logits.append(logit)
        else:
            filtered_logits.append(-math.inf)

    next_token_id = torch.argmax(torch.tensor(filtered_logits)).item()
    next_token_text = model.decode([next_token_id])


    print("current:", current_text)
    print("next_token_id:", next_token_id)
    print("decoded:", repr(next_token_text))

if __name__ == "__main__":
    main()

# ✅ Step 1：项目初始化
# uv sync
# ✅ Step 2：安装本地 SDK
# uv pip install -e ./llm_sdk
# ✅ Step 3：安装依赖
# uv add pydantic transformers huggingface-hub
# ✅ Step 4：安装 CPU 版 torch（避免 CUDA 巨大包）
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu
# ⚠️ Step 5（最关键！）：磁盘空间问题

# 你刚刚的问题：

# 模型 1.5GB
# /home 只有 1GB 空间 ❌
# ✅ 解决方案（考试必用）
# mkdir -p /goinfre/$USER/huggingface

# export HF_HOME=/goinfre/$USER/huggingface
# export TRANSFORMERS_CACHE=/goinfre/$USER/huggingface

# 👉 永久写入：

# echo 'export HF_HOME=/goinfre/$USER/huggingface' >> ~/.zshrc
# echo 'export TRANSFORMERS_CACHE=/goinfre/$USER/huggingface' >> ~/.zshrc
# source ~/.zshrc
# ✅ Step 6：验证模型
# uv run python -c "from llm_sdk import Small_LLM_Model; print('OK')"
# ✅ Step 7：运行测试
# uv run python src/llm_test.py
# 🧠 Step 8：如果报错（标准排查）
# 空间问题：
# df -h
# 清缓存：
# uv cache clean
# rm -rf ~/.cache/huggingface
# rm -rf ~/.cache/pip