import re
from typing import Any, cast


def extract_numbers(prompt: str) -> list[float]:
    """Extract numbers from a prompt."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    return [float(match) for match in matches]


def extract_quoted_strings(prompt: str) -> list[str]:
    """Extract all texts inside single or double quotes."""
    matches = re.findall(r"(['\"])(.*?)\1", prompt)
    return [match[1] for match in matches]


def extract_greet_name(prompt: str) -> str:
    """Extract the name after the word greet."""
    match = re.search(
        r"greet\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # * >=0, + >= 1
        prompt,
        re.IGNORECASE,
    )
    return cast(str, match.group(1)) if match else ""


def extract_string_parameter(
    prompt: str,
    parameter_name: str,
    quoted: list[str]
) -> str:
    """Extract a string value using the schema parameter role."""
    lower = prompt.lower()
    name = parameter_name.lower()

    if name == "regex":
        if (
            "numbers" in lower
            or "digits" in lower
            or "number" in lower
            or "digit" in lower
        ):
            return r"\d+"
        elif "vowels" in lower or "vowel" in lower:
            return r"[aeiouAEIOU]"
        elif (
            "separators" in lower
            or "commas" in lower
            or "separator" in lower
            or "comma" in lower
        ):
            return ","
        elif quoted:
            return re.escape(quoted[0])
            # re.escape() for special characters，e.g. .、*、?。
            # if no numbers / vowels，but ""，transfer the first "" as regex。
        return ""

    if name == "replacement":
        if "asterisks" in lower:
            return "*"
        if "spaces" in lower or "space" in lower:
            return " "
        if "with" in lower and quoted:
            if len(quoted) >= 3:
                return quoted[1]
        match = re.search(r"with\s+['\"]?([A-Za-z0-9_*]+)['\"]?", prompt)
        return match.group(1) if match else ""

    if name == "source_string":
        return quoted[-1] if quoted else ""

    if name == "s":
        return quoted[0] if quoted else ""

    if name == "name":
        return extract_greet_name(prompt)

    return quoted[0] if quoted else ""


def extract_boolean(prompt: str) -> bool:
    """Extract a boolean-like value from a prompt."""
    lower = prompt.lower()

    true_words = ["true", "yes", "enable", "enabled", "on"]
    false_words = ["false", "np", "disable", "disabled", "off"]

    for word in false_words:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return False

    for word in true_words:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return True

    return False


def build_args(
    prompt: str,
    function_schema: dict[str, Any],
) -> dict[str, Any]:
    """Build arguments using schema/type-driven extraction."""
    raw_parameters = function_schema.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        return {}
    # "parameters": {
    #   "s": {
    #     "type": "string"
    #   }
    # },

    numbers = extract_numbers(prompt)
    quoted = extract_quoted_strings(prompt)
    result: dict[str, Any] = {}
    number_index = 0

    for parameter_name, parameter_schema in raw_parameters.items():
        parameter_type = ""

        if isinstance(parameter_schema, dict) and "enum" in parameter_schema:
            enum_values = parameter_schema.get("enum")

            if isinstance(enum_values, list) and enum_values:
                lower = prompt.lower()

                for value in enum_values:
                    if str(value).lower() in lower:
                        result[parameter_name] = value
                        break
                else:
                    result[parameter_name] = enum_values[0]

                continue

        if isinstance(parameter_schema, dict):
            parameter_type = str(parameter_schema.get("type", ""))

        if parameter_type == "number":
            result[parameter_name] = (
                numbers[number_index]
                if number_index < len(numbers)
                else 0.0
            )
            number_index += 1
        elif parameter_type == "integer":
            result[parameter_name] = (
                int(numbers[number_index])
                if number_index < len(numbers)
                else 0
            )

        elif parameter_type == "boolean":
            result[parameter_name] = extract_boolean(prompt)

    # "parameters": {
    #   "a": {
    #     "type": "number"
    #   },
    #   "b": {
    #     "type": "number"
    #   }
    # },

        elif parameter_type == "string":
            result[parameter_name] = extract_string_parameter(
                prompt=prompt,
                parameter_name=str(parameter_name),
                quoted=quoted,
            )
        else:
            result[parameter_name] = ""
    return result
