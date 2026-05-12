import json
import math
from enum import Enum
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class JsonState(Enum):
    """States used to generate the function-call JSON object."""
    START = "start"
    PROMPT = "prompt"
    NAME = "name"
    PARAMETERS = "parameters"
    DONE = "done"


# only use Enum defined members
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


class SchemaConstrainedDecoder:
    """Generate function-call JSON with state and schema constraints."""
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self.vocabulary = VocabularyMap(model)
        self.valid_token_cache: dict[str, list[tuple[int, str]]] = {}

    def choose_token_for_expected_text(
        self,
        current: str,
        expected_remaining: str,
    ) -> str:
        """Choose the best valid token for the current state fragment."""
        input_ids = self.model.encode(current)[0].tolist()
        if not input_ids:
            input_ids = self.model.encode(" ")[0].tolist()

        logits = self.model.get_logits_from_input_ids(input_ids)
        # token_id -> next-token score
        valid_tokens = self.valid_token_cache.get(expected_remaining)
        # valid_tokens = rule filtering result:
        # from vocabulary filter out legal tokens
        # logits = model scoring result of filtered tokens:
        # changes when context changes

        if valid_tokens is None:
            valid_tokens = [
                (token_id, token_text)
                for token_id, token_text in self.vocabulary.decoded_tokens
                if expected_remaining.startswith(token_text)
            ]
            self.valid_token_cache[expected_remaining] = valid_tokens

        best_token = ""
        best_score = -math.inf

        for token_id, token_text in valid_tokens:
            score = float(logits[token_id])
            if score > best_score:
                best_score = score
                best_token = token_text
        if best_token == "":
            raise RuntimeError(
                "No valid token found for expected text: "
                f"{expected_remaining[:40]!r}"
                # print the first 40 chars  !r in the form of repr/debug
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

    def json_value_for_schema(self, value: Any, parameter_schema: Any) -> Any:
        """Serialize one parameter value acoording to its schema type."""
        parameter_type = ""

        if isinstance(parameter_schema, dict):
            parameter_type = str(parameter_schema.get("type", ""))

        enum_values = parameter_schema.get("enum")
        if (isinstance(enum_values, list) and enum_values):
            if value in enum_values:
                return value
            return enum_values[0]

        if parameter_type == "number":
            if isinstance(value, (int, float)):
                return float(value)

        if parameter_type == "string":
            return str(value)

        if parameter_type == "boolean":
            return bool(value)

        return value

    def build_clean_parameters(
            self,
            function_schema: dict[str, Any] | None,
            parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Build parameters that follow the selected schema."""
        if function_schema is None:
            return {}

        raw_parameter_schemas = function_schema.get("parameters", {})

        if not isinstance(raw_parameter_schemas, dict):
            return {}

        clean_parameters: dict[str, Any] = {}

        for parameter_name, parameter_schema in raw_parameter_schemas.items():
            clean_parameters[str(parameter_name)] = self.json_value_for_schema(
                parameters.get(str(parameter_name)),
                parameter_schema,
            )

        return clean_parameters

    def constrained_value(
            self,
            current: str,
            value: Any,
    ) -> str:
        """Generate one serialized valie with constrained decoding."""
        expected_text = json.dumps(value)
        return self.generate_expected_text(current=current,
                                           expected_text=expected_text)

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

        clean_parameters = self.build_clean_parameters(
            function_schema=function_schema,
            parameters=parameters,
        )

        while state != JsonState.DONE:
            if state == JsonState.START:
                current += '{"prompt": '
                state = JsonState.PROMPT

            elif state == JsonState.PROMPT:
                current = self.constrained_value(current, prompt)
                current += ', "name": '
                state = JsonState.NAME

            elif state == JsonState.NAME:
                current = self.constrained_value(current, function_name)
                current += ', "parameters": '
                state = JsonState.PARAMETERS

            elif state == JsonState.PARAMETERS:
                current = self.constrained_value(current, clean_parameters)
                current += "}"
                state = JsonState.DONE

        json.loads(current)  # json str -> python object: FunctionCall
        return current
