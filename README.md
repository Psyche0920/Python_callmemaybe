# Call Me Maybe
*This project was created as part of the 42 curriculum by wehan.*

## Description
Call Me Maybe is a function-calling system that converts natural language prompts into structured JSON function calls using a small language model.

The project combines:
- LLM-based function selection
- schema-driven parameter extraction
- constrained decoding
- state-machine JSON generation
- JSON validation

to produce valid structured output.

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

### Features

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

### Project Structure

```text
.
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
│
├── llm_sdk/
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── decoder.py
│   ├── extracter.py
│   ├── main.py
│   ├── models.py
│   ├── parser.py
│   ├── selector.py
│   └── utils.py
│
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock

```

---

## Instructions

### Installation

Install dependencies with:

```bash
uv sync
```

The `llm_sdk/` directory must exist at the repository root.

---

## Example Usage

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

## Algorithm

### Function Selection

The selector builds a prompt containing:

- available functions
- descriptions
- user request managing limited storage environments on 42 Linux machines

The model performs one forward pass and scores candidate functions using logits from the first distinguishing token after the shared `fn` prefix.
 managing limited storage environments on 42 Linux machines
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

Token filtering is constrained by deterministic expected-text generation rather than free-form sampling.

This produces structurally valid JSON output.

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
- regex extraction

Accuracy: 11/11 test cases passing (100%)

- Function selection: 100%
- Number extraction: 100%
- String extraction: 100%
- Regex generation: 100%
- JSON validity: 100%

Complexity:
- Linear in input size
- Single model load
- Lightweight per-prompt inference

Reliability:
- Structurally valid JSON output via deterministic state-machine generation and constrained token selection.

---


---

## Design Decisions

Key implementation choices include:
- using logits-based function selection instead of prompt-only matching
- combining deterministic state-machine generation with constrained token filtering
- using schema-aware parameter extraction for different parameter types
- validating structured outputs with Pydantic and json.loads()

---

## Challenges Faced

Main implementation challenges included:
- maintaining valid JSON structure during constrained decoding
- handling tokenizer vocabulary boundaries during token filtering
- balancing deterministic generation with model-based scoring
- managing limited storage environments on 42 Linux machines

---

## Testing

Testing with:
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

---

### Note for 42 machines

If `uv sync` fails because of limited `/home` storage, redirect the uv cache and virtual environment to `/goinfre`:

```bash
export UV_CACHE_DIR=/goinfre/$USER/uv_cache
export HF_HOME=/goinfre/$USER/hf_cache
export PIP_CACHE_DIR=/goinfre/$USER/pip_cache
export TMPDIR=/goinfre/$USER/tmp
export UV_PROJECT_ENVIRONMENT=/goinfre/$USER/venvs/python-callmemaybe
export UV_LINK_MODE=copy
uv sync --python 3.11
```