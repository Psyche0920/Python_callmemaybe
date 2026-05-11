*This project has been created as part of the 42 curriculum by wehan.*

# Call Me maybe

## Description
Call Me Maybe is a function_calling tool that translates natural language prompts into structured JSON function calls.

Instead of directly answering a user request, the program selects the most suitable function from a provided function definition file and extracts the required arguments according to the funcion schema.

Example:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}

The project focuses on reliable structured output genreation with a small language model. Since small models may not always produce valid JSON by prompting alone, this implementation uses constrained decoding to guide JSON generation.

## Project Structure
.
├── src/
│   ├── __main__.py
│   ├── main.py
│   ├── selector.py
│   ├── extractor.py
│   ├── decoder.py
│   ├── parser.py
│   ├── utils.py
│   └── models.py
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── llm_sdk/
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md

# Installation
The project uses uv.
uv sync
The llm_sdk/ directory must be present at the root of the repository.

# Usage
Run the program with the default input and output paths:
uv run python -m src

Run with custom files:
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
  The output directory is created automatically during execution.

# Input Files
functions_definition.json
This file contains the available funcions, their descriptions, parameters and return types.
Example:
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }
]

function_calling_tests.json
This file contains the prompts to process.
Example:

[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]

# Output Format
The progra writes a JSON array to:
data/output/function_calling_results.json

Each entry follows this schema:
{
  "prompt": "Original prompt",
  "name": "function_name",
  "parameters": {}
}

The output contains no extra prose, comments, or text outside the JSON structure.

## Algorithm Explanation
The pipleline has four main steps.

1. Function Selection
Function selection is performed with small LLM through logits scoring.

The program builds a context containing:
-the available function names
-their descriptions
-the user prompt
-a prefix: The correct function is: fn
The model then provides logits for the next token. Each function name is scored by checking the model's preference for the first token after the shared fn prefix.

The scores are converted into probabilities with softmax. If the best probability is too low, the algorithm treats the prompt as ambiguous and returns an empty function call.

This keeps function selection LLM-based while avoiding pure keyword matching.

2. Parameter Extraction
Argument extraction is schema-driven and uses regex-based text parsing.
The program reads the selected function schema and extracts arguments according to parameter type:
-number: extracts numeric values
-integer: extracts numbers and converts them to integers
-string: extracts quoted strings or named values
-boolean: detects words such as enable, disable, on, and off
-enum: only allows values listed in the schema

For example, if a schema contains:
"firmware": {
  "type": "string",
  "enum": ["stable", "beta", "nightly"]
}
the output will only contain of these allowed values.

3. Constrained JSON Generation
The JSON output is generated with a state machine.

The decoder moves through states such as:
-object start
-prompt key
-prompt value
-name key
-name value
-parameters key
-parameters keys and values
-object close

At each step, the decoder knows the exact JSON fragment that should be generated next.

The model logits are used to choose the best valid token. Tokens that do not match the expected JSON fragment are not allowed to participate in selection. This is equivalent to setting invalid token logits to negative infinity. ???

This guarantees that the generated text remains valid JSON.

4. Validation and Writing
After generation:
-json.loads() checks that the generated JSON is parseable
-Pydantic validates the final FunctionCall obejct.
-json.dump() writes the clean JSON array to the output file.

## Design Decisions
# LLM for Function Selection
The function name is selected using LLM logits, not hardcoded keyword rules. This follows the project requirement that the function should be chosen by the model.


# Schema-Driven Arguments
The parameters are extracted according to the function schema instead of fixed examples.
This makes the program more flexible when the input function definition change.

# State machine for JSON
A state machineis used to make JSON generation explicit and reliable. This avoids relying on the model to spontaneously produce valid JSON.

## Graceful Error Handling
Missing files, invalid JSON files, malformed inputs, and processing errors are handled with clear messages. The program avoids unexpected crashes and keeps the output format valid whenever possible.

## Performance and Reliability
The selector uses one model forward pass per prompt for function scoring. This keeps runtime practical and helps the program stay within the expected evaluation time.

The constrained decoder ensures that output JSON is always parseable.

The implementation was tested with:
-simple arthmetic prompts
-negative numbers
-decimal numbers
-large numbers
-string reversal
-empty strings
-regex substitution
-enum values
-boolean values
-integer parameters
-ambiguous prompts
-missing or malformed input files

## Testing Strategy
Run the project normally
uv run python -m src

Test missing input file:
uv run python -m src --input data/input/not_found.json

Expected behavior:
Error: file not found: data/input/not_found.json

Test invalid JSON:
echo "{ bad json" > data/input/bad.json
uv run python -m src --input data/input/bad.json

Expected behavior:
Error: invalid JSON file: data/input/bad.json

Run linting:
make lint

Run strict linting:
make lint-strict

## Makefile
The Makefile provides common commands:
make install
make run
make debug
make clean
make lint
make lint-strict

## Challenges Faced
# Keeping JSON Valid
Small language models are unreliable when asked to directly output JSON. The solution was to guide generation with a state machine and only allow valid tokens at each step.

# Balancing Accuracy and Runtime
Scoring full function names token by token can be more accurate but slower. The implementation uses a faster logits-based selection strategy to keep runtime reasonale.

# Handling Different Parameter Types
The input function definitions may change during evaluation, so the extractor supports multiple schema types, including numbers, strings, booleans, integers, and enums.

# Avoiding Crashes
Input files may be missing or malformed. File loading and prompt processing are wrapped with error handing so the program exits cleanly or produces safe fallback output.

## AI Usage
AI tools were used to help reason about architecture, edge cases, README wording, and testing strategies.

All code was reviewed and adapted manually. The final implementation was tested and understood by the author.

Resources:
LLM Model Introduction: https://www.bilibili.com/video/BV11nSjB2ErQ/?spm_id_from=333.1391.0.0&p=6&vd_source=13ca61368de7608ca77ae75e59d223d4



