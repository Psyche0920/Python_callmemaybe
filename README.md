# Call Me Maybe

*This project was created as part of the 42 curriculum by wehan.*

## Overview

Call Me Maybe is a lightweight function-calling system that converts natural-language prompts into structured JSON function calls using a small language model.

The project combines:

- LLM-based function selection
- schema-driven parameter extraction
- constrained decoding
- state-machine JSON generation
- JSON validation

to guarantee valid structured output.

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
```

---

## Features

- LLM logits-based function selection
- schema-aware argument extraction
- constrained JSON decoding
- deterministic state-machine generation
- valid JSON output
- graceful error handling
- support for:
  - numbers
  - integers
  - booleans
  - strings
  - enums
  - regex-based substitutions

---

## Project Structure

```text
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
├── llm_sdk/
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Installation

Install dependencies with:

```bash
uv sync
```

The `llm_sdk/` directory must exist at the repository root.

---

## Usage

Run with default paths:

```bash
uv run python -m src
```

Run with custom files:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

---

## How It Works

### Function Selection

The selector builds a prompt containing:

- available functions
- descriptions
- user request

The model performs one forward pass and scores candidate functions using logits from the first distinguishing token after the shared `fn` prefix.

Softmax converts scores into probabilities.

Low-confidence prompts return:

```json
{
  "name": "",
  "parameters": {}
}
```

---

### Parameter Extraction

Arguments are extracted according to the selected schema.

Supported parameter types:

- `number`
- `integer`
- `string`
- `boolean`
- `enum`

Regex-based parsing is used for:

- numbers
- quoted strings
- substitutions
- booleans
- regex patterns

---

### Constrained Decoding

JSON generation is guided by a deterministic state machine.

Only tokens matching the expected JSON fragment are allowed during decoding.

This guarantees structurally valid JSON output.

---

### Validation

Generated outputs are validated with:

- `json.loads()`
- `Pydantic`

and written using:

- `json.dump()`

---

## Performance

The implementation is optimized for runtime stability using:

- one forward pass per prompt
- first-token function scoring
- valid-token caching
- lightweight regex extraction

---

## Testing

Tested with:

- arithmetic prompts
- negative and decimal numbers
- regex substitutions
- enum values
- booleans
- empty strings
- malformed files
- ambiguous prompts

Run linting:

```bash
make lint
```

Run strict linting:

```bash
make lint-strict
```

---

## Makefile Commands

```bash
make install
make run
make debug
make clean
make lint
make lint-strict
```

---

## AI Usage

AI tools were used for:

- architecture discussion
- edge-case analysis
- README refinement
- testing ideas

All code was manually reviewed, adapted, and understood by the author.

---

## Sources

- LLM Introduction (CN): https://www.bilibili.com/video/BV11nSjB2ErQ/
- ML Introduction (CN): https://www.bilibili.com/video/BV1SjDsBLEzN/
- Logits: https://en.wikipedia.org/wiki/Logit
- UTF-8: https://en.wikipedia.org/wiki/UTF-8
- Softmax: https://en.wikipedia.org/wiki/Softmax_function