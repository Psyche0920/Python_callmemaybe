import json
import math
from enum import Enum
from typing import Any

from llm_sdk import Small_LLM_Model


class JsonState(Enum):
    """States used to generate the function-call JSON object."""
    START = "start"
    PROMPT_KEY = "prompt_key"
    PROMPT_VALUE = "prompt_value"
    NAME_KEY = "name_key"
    NAME_VALUE = "name_value"
    PARAMETERS_KEY = "parameters_key"
    PARAMETERS_OPEN = "parameters_open"
    PARAMETER_KEY = "parameter_key"
    PARAMETER_VALUE = "parameter_value"
    PARAMETERS_CLOSE = "parameters_close"
    OBJECT_CLOSE = "object_close"    
    DONE = "done"


# 你只能使用 Enum 里“已经定义过的成员”，否则程序直接报错
# otherwise silent error and crash


class VocabularyMap:
    """Public vocabulary/token map used by the constrained decoder."""
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self.token_ids = self._load_token_ids()
        self.decoded_tokens = self._decode_tokens()
    
    def _load_token_ids(self) -> list[int]:
        """Load public token ids from the SDK vocabulary JSON file."""
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        if isinstance(data, dict):
            ids = [int(value) for value in data.values()]
        elif isinstance(data, list):
            ids = list(range(len(data)))
        else:
            ids = []
        
        return sorted(set(ids))
    
    def _decode_tokens(self) -> list[tuple[int, str]]:
        """Decode each public vocabulary token id once."""
        decoded: list[tuple[int, str]] = []
        for token_id in self.token_ids:
            try:
                token_text = self.model.decode([token_id])
            except Exception:
                continue
            if token_text:
                decoded.append((token_id, token_text))
        return decoded
    #     VocabularyMap：
    # 👉 “把 vocab 转换成 constrained decoder 可直接使用的 tokenizer map”
    # 1️⃣ 读取 vocab file
    # json.load(file)
    # 2️⃣ 统一格式
    # dict/list
    # ↓
    # token ids
    # 3️⃣ 用 tokenizer 真正 decode
    # token id
    # ↓
    # model.decode()
    # ↓
    # token text
    # 🎯 最终得到：
    # [
    #     (token_id, decoded_token_text)
    # ]
    # 这才是 constrained decoder 真正需要的数据结构。


class SchemaConstrainedDecoder:
    """Generate function-call JSON with state and schema constraints."""
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self.vocabulary = VocabularyMap(model)
    
    def choose_token_for_expected_text(
        self,
        current: str,
        expected_remaining: str,
    ) -> str:
        """Choose the best valid token for the current state fragment."""
        input_ids = self.model.encode(current)[0].tolist()
        if not input_ids:
            input_ids=self.model.encode(" ")[0].tolist()

        logits = self.model.get_logits_from_input_ids(input_ids)
        # token_id -> next-token score
        best_token = ""
        best_score = -math.inf

        for token_id, token_text in self.vocabulary.decoded_tokens:
            if expected_remaining.startswith(token_text):  # to find the best within legal tokens
                score = float(logits[token_id])
                if score > best_score:
                    best_score = score
                    best_token = token_text
        if best_token == "":
            raise RuntimeError(
                "No valid token found for expected text: "
                f"{expected_remaining[:40]!r}"  # print the first 40 chars  !r 用 repr/debug 形式打印
            )
        return best_token
    
    def generate_expected_text(self, current: str, expected_text: str) -> str:
        """Generate one state-specific fragment via token filtering."""
        generated = ""
        while generated != expected_text:
            remaining = expected_text[len(generated):]
            next_token = self.choose_token_for_expected_text(
                current=current + generated,
                expected_remaining=remaining,
            )
            generated += next_token
        return current + generated
# expected_text - current = expected_remaining
#  expected_text
# ↓
# 作为“合法目标”
# ↓
# 模型输出 logits
# ↓
# 遍历 vocab
# ↓
# 只允许合法 token
# ↓
# 选最高分 token
# ↓
# 一步步生成 JSON

    def json_value_for_schema(self, value: Any, parameter_schema: Any) -> str:
        """Serialize one parameter value acoording to its schema type."""
        parameter_type = ""
        if isinstance(parameter_schema, dict):
            parameter_type = str(parameter_schema.get("type", ""))
        
        if parameter_type == "number":
            if isinstance(value, int | float):
                return json.dumps(float(value))
        
        if parameter_type == "string":
            return json.dumps(str(value))
        
        return json.dumps(value)

    def generate_function_call_json(
        self,
        prompt: str,
        function_name: str,
        function_schema: dict[str, Any] | None,
        parameters: dict[str, Any],
    ) -> str:
        """Generate a valid function-call JSON object state by state."""
        current = ""
        state = JsonState.START
        parameter_index =  0

        raw_parameter_schemas: Any = {}
        if function_schema is not None:
            raw_parameter_schemas = function_schema.get("parameters", {})

        parameter_schemas: dict[str, Any]
        if isinstance(raw_parameter_schemas, dict):
            parameter_schemas = raw_parameter_schemas
        else:
            parameter_schemas = {}
        
        parameter_names = list(parameter_schemas.keys())

        while state != JsonState.DONE:
            if state == JsonState.START:
                current = self.generate_expected_text(current, "{")
                state = JsonState.PROMPT_KEY

            elif state == JsonState.PROMPT_KEY:
                current = self.generate_expected_text(current, '"prompt": ')
                state = JsonState.PROMPT_VALUE

            elif state == JsonState.PROMPT_VALUE:
                current = self.generate_expected_text(current, json.dumps(prompt))
                current = self.generate_expected_text(current, ", ")
                state = JsonState.NAME_KEY
            
            elif state == JsonState.NAME_KEY:
                current = self.generate_expected_text(current, '"name": ')
                state = JsonState.NAME_VALUE

            elif state == JsonState.NAME_VALUE:
                current = self.generate_expected_text(current, json.dumps(function_name))  # python object -> json str
                current = self.generate_expected_text(current, ", ")
                state = JsonState.PARAMETERS_KEY

            elif state == JsonState.PARAMETERS_KEY:
                current = self.generate_expected_text(current, '"parameters": ')
                state = JsonState.PARAMETERS_OPEN

            elif state == JsonState.PARAMETERS_OPEN:
                current = self.generate_expected_text(current, "{") 
                if parameter_names:
                    state = JsonState.PARAMETER_KEY
                else:
                    state = JsonState.PARAMETERS_CLOSE

            
            elif state == JsonState.PARAMETER_KEY:
                parameter_name = parameter_names[parameter_index]
                current = self.generate_expected_text(
                    current,
                    json.dumps(parameter_name)
                )
                current = self.generate_expected_text(current, ": ")
                state = JsonState.PARAMETER_VALUE
            
            elif state == JsonState.PARAMETER_VALUE:
                parameter_name = parameter_names[parameter_index]
                parameter_schema = parameter_schemas[parameter_name]
                value_text = self.json_value_for_schema(
                    parameters.get(parameter_name),
                    parameter_schema,
                )
                current = self.generate_expected_text(current, value_text)
                
                parameter_index += 1
                if parameter_index < len(parameter_names):
                    current = self.generate_expected_text(current, ", ")
                    state = JsonState.PARAMETER_KEY
                else:
                    state = JsonState.PARAMETERS_CLOSE
                
            elif state == JsonState.PARAMETERS_CLOSE:
                current = self.generate_expected_text(current, "}")
                state = JsonState.OBJECT_CLOSE

            elif state == JsonState.OBJECT_CLOSE:
                current = self.generate_expected_text(current, "}")
                state = JsonState.DONE

        json.loads(current)  # json str -> python object: FunctionCall
        return current

# e.g. 
# {
#     "prompt": "What is the sum of 2 and 3?",
#     "fn_name": "fn_add_numbers",
#     "parameters": {
#       "a": 2.0,
#       "b": 3.0
#     }
#   }



# 1. 什么是 function-call object？

# 就是最终输出里的每一条 JSON object。

# 比如：

# {
#   "prompt": "What is the sum of 2 and 3?",
#   "name": "fn_add_numbers",
#   "parameters": {
#     "a": 2.0,
#     "b": 3.0
#   }
# }

# 这个整体就是一个 function-call object。

# 意思是：

# 自然语言 prompt
# ↓
# 被转换成一个可执行的函数调用描述

# 它不是直接回答：

# 5

# 而是告诉系统：

# 调用 fn_add_numbers
# 参数是 a=2.0, b=3.0

# 所以：

# FunctionCall(BaseModel)

# 就是这个 JSON object 的 Python/Pydantic 版
# 2. 
# Python object/data

# ↓
# serialize
# (json.dumps)

# ↓

# JSON text
# (Python str containing JSON syntax)

# ↓

# tokenization
# (model.encode)

# ↓

# token ids

# ↓

# LLM next-token generation
# (logits + constrained decoding)

# ↓

# generated token ids

# ↓

# decode
# (model.decode)

# ↓

# JSON text

# ↓

# deserialize
# (json.loads)

# ↓

# Python object