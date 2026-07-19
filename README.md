# AI API Learning Project

A day-by-day Python project for learning AI APIs — from first API calls and parameter tuning, to tokens, streaming, stop sequences, structured JSON with Pydantic validation, multi-turn chat memory strategies, chain-of-thought prompting, and building tool-using agents with dispatchers and structured logging.

**Primary provider:** Groq (free tier) · **Model:** `llama-3.3-70b-versatile`

---

## Project Structure

```
day01/
├── Day_1_main.py                        # Multi-provider client (OpenAI + Anthropic)
├── Day_1_test_setup.py                  # Test API connections
├── Day_1_ten_run_with_diff_prams.py     # 10 Groq calls with different parameters
├── Day_2_gork_token_counter.py          # Token counting & cost estimation
├── Day_2_gorq_steaming.py               # Real-time streaming responses
├── Day_2_stop_sequence.py               # Stop sequence experiments
├── Day_3_json_formatted_response.py     # Structured JSON output from Groq
├── Day_3_pedantic_validation_and_retry.py # Pydantic validation with retries
├── Day_4_multiturn_conversational_loop.py # Multi-turn chat with memory strategies
├── Day_5_chain_of_thought_(COT).py      # Chain-of-thought prompting experiment
├── Day_8_tools_as_Json_schemas.py       # First tool-use loop (JSON schemas)
├── Day_9_tool_use_loop.py               # Multi-tool agent loop (calculator, time, jokes)
├── Day_10_tool_use_simulteniously.py    # Simultaneous multi-tool calls + web search
├── Day_11_tool_use_read_write.py        # Context Q&A with web search fallback
├── day_12_logger_and_dispatcher.py      # Tool dispatcher registry + structured logging
├── all_tool_definition.py               # Shared tool JSON schemas
├── sandbox/                             # Local files for read/write demos (todo.txt, etc.)
├── requirements.txt
├── .env                                 # API keys (create locally — not committed)
├── .gitignore
├── README.md
└── venv01/                              # Virtual environment (create locally)
```

---

## Prerequisites

- **Python 3.10+** (tested with Python 3.12)
- **pip**
- **Groq API key** (free) — required for most scripts

```bash
python3 --version
```

---

## Environment Setup

### 1. Navigate to the project

```bash
cd /path/to/day01
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv01
source venv01/bin/activate        # Linux / macOS
# venv01\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package         | Purpose                              | Used from   |
|-----------------|--------------------------------------|-------------|
| `groq`          | Groq API client (primary)            | Day 1–12    |
| `python-dotenv` | Load `.env` variables                | All days    |
| `requests`      | HTTP (web search, jokes, weather)    | Day 9–12    |
| `tiktoken`      | Token counting                       | Day 2, 4    |
| `pydantic`      | Schema validation                    | Day 3       |
| `openai`        | OpenAI API client                    | Day 1       |
| `anthropic`     | Anthropic (Claude) API client        | Day 1       |
| `google-genai`  | Google Gemini API client             | Day 1       |
| `tenacity`      | Retry/backoff utilities (available)  | Optional    |

### 4. Create `.env`

**Never commit this file.**

```env
# Free-tier (recommended — required for Day 1–12 Groq scripts)
GORQ_API_KEY=gsk-your-groq-key-here

# Optional — better web search in Day 10–12 (falls back to DuckDuckGo if unset)
SERPER_API_KEY=your-serper-key-here

# Optional — paid or quota-limited
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GEMINI_API_KEY=your-gemini-key
```

> **Note:** This project uses `GORQ_API_KEY` (typo). Most scripts also accept `GROQ_API_KEY`.

| Provider    | Get a key                                              | Cost      |
|-------------|--------------------------------------------------------|-----------|
| **Groq**    | https://console.groq.com                               | Free tier |
| **Gemini**  | https://aistudio.google.com/apikey                     | Free tier |
| **OpenAI**  | https://platform.openai.com/api-keys                   | Paid      |
| **Anthropic** | https://console.anthropic.com/settings/keys          | Paid      |

### 5. Update `requirements.txt` (optional)

```bash
pip install some-package
pip freeze > requirements.txt
```

---

## Quick Start (Groq only)

```bash
cd day01
python3 -m venv venv01
source venv01/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
GORQ_API_KEY=gsk-your-key-here
```

Run day-by-day in order (see sections below).

---

# Day 1 — Setup, Providers & Parameters

**Goals:** Connect to AI APIs, test keys, experiment with temperature, max tokens, and system prompts.

| File | Command |
|------|---------|
| Test connection | `python3 Day_1_test_setup.py` |
| Parameter experiments | `python3 Day_1_ten_run_with_diff_prams.py` |
| OpenAI + Anthropic demo | `python3 Day_1_main.py` |

---

### `Day_1_test_setup.py` — Test API connections

Quick health check for API keys. Runs **Groq** by default.

```bash
python3 Day_1_test_setup.py
```

| Function           | Provider  | Key required        | Default   |
|--------------------|-----------|---------------------|-----------|
| `test_groq()`      | Groq      | `GORQ_API_KEY`      | Active    |
| `test_gemini()`    | Gemini    | `GEMINI_API_KEY`    | Available |
| `test_openai()`    | OpenAI    | `OPENAI_API_KEY`    | Available |
| `test_anthropic()` | Anthropic | `ANTHROPIC_API_KEY` | Available |

To test another provider, call its function in the `if __name__ == "__main__"` block.

---

### `Day_1_ten_run_with_diff_prams.py` — Parameter experiments

Sends **10 API calls** to Groq with different settings. Type `y` when prompted.

```bash
python3 Day_1_ten_run_with_diff_prams.py
```

| Call | What it tests                                |
|------|----------------------------------------------|
| 1    | Baseline (temperature 1.0)                   |
| 2    | Low temperature 0.0 (deterministic)          |
| 3    | High temperature 1.5 (creative)              |
| 4    | Short response (`max_tokens=20`)             |
| 5    | Long response (`max_tokens=300`)             |
| 6    | System prompt: AI expert                     |
| 7    | System prompt: teacher for kids              |
| 8    | System prompt: creative writer               |
| 9    | Multi-turn conversation                      |
| 10   | High temperature + system prompt + long output |

#### Temperature quick reference

| Range    | Behavior                           | Use for                |
|----------|------------------------------------|------------------------|
| 0.0–0.3  | Focused, consistent, repeatable  | Facts, code, summaries |
| 0.5–0.9  | Balanced (good default)            | Chat, explanations     |
| 1.2–2.0  | Creative, varied, unpredictable  | Writing, brainstorming |

---

### `Day_1_main.py` — Multi-provider AI client

Demonstrates a reusable `AIClient` class for **OpenAI** and **Anthropic** (paid — requires credits).

```bash
python3 Day_1_main.py
```

- OpenAI: `gpt-4`
- Anthropic: `claude-3-opus-20240229`

---

# Day 2 — Tokens, Streaming & Stop Sequences

**Goals:** Count tokens, stream responses in real time, control where the model stops generating.

| File | Command |
|------|---------|
| Token counter | `python3 Day_2_gork_token_counter.py` |
| Streaming | `python3 Day_2_gorq_steaming.py` |
| Stop sequences | `python3 Day_2_stop_sequence.py` |

---

### `Day_2_gork_token_counter.py` — Token counting & cost estimate

Estimates tokens **before** the API call, then compares with actual usage.

```bash
python3 Day_2_gork_token_counter.py
```

- Uses `tiktoken` for estimates
- Runs 3 test calls (simple, system prompt, multi-turn)
- Compares estimated vs actual tokens and cost
- Prints summary report

---

### `Day_2_gorq_steaming.py` — Streaming responses

Streams Groq responses in real time with chunk-level metrics.

```bash
python3 Day_2_gorq_steaming.py
```

| Case | Description                  |
|------|------------------------------|
| 1    | Basic streaming              |
| 2    | Streaming with system prompt |
| 3    | Long-form content streaming  |
| 4    | Colored chunk streaming      |

**Metrics:** chunks received, estimated tokens, time to first chunk, chunks/sec.

> **Note:** Stream pieces are **chunks**, not tokens. Token count is estimated after the full response arrives.

---

### `Day_2_stop_sequence.py` — Stop sequence experiments

Runs **7 tests** showing how `stop=` controls where generation ends.

```bash
python3 Day_2_stop_sequence.py
```

| Test | Stop sequence        | Purpose                          |
|------|----------------------|----------------------------------|
| 1    | None                 | Baseline — stops at `max_tokens` |
| 2    | `["\n"]`             | Stop at newline                  |
| 3    | `["."]`              | Stop at period                   |
| 4    | `["\n\n", "!", "?"]` | Multiple stop options            |
| 5    | `["}\n"]`            | JSON object boundary             |
| 6    | `["\n\n"]`           | Stop at paragraph break          |
| 7    | `["."]`              | Fill-in-the-blank sentence       |

**Key concept:** `finish_reason: "stop"` = stop sequence hit · `finish_reason: "length"` = max tokens hit.

---

# Day 3 — Structured JSON & Validation

**Goals:** Get reliable JSON from the model, parse it, and validate with Pydantic schemas.

| File | Command |
|------|---------|
| JSON formatting | `python3 Day_3_json_formatted_response.py` |
| Pydantic + retry | `python3 Day_3_pedantic_validation_and_retry.py` |

---

### `Day_3_json_formatted_response.py` — Structured JSON output

Gets fictional person data as JSON from Groq and parses it.

```bash
python3 Day_3_json_formatted_response.py
```

**Features:**

- Groq JSON mode: `response_format={"type": "json_object"}`
- `extract_json()` — strips markdown ` ```json ` fences before parsing
- Runs two identical requests to show consistent output

**Example output:**

```json
{
  "name": "Ethan Thompson",
  "age": 32,
  "city": "New York",
  "hobby": "Playing Guitar"
}
```

---

### `Day_3_pedantic_validation_and_retry.py` — Pydantic validation with retries

Gets JSON from Groq, validates against a `Person` schema, and retries up to 3 times with error feedback.

```bash
python3 Day_3_pedantic_validation_and_retry.py
```

**Flow:**

```
Groq API → extract_json() → Person (Pydantic) → success or retry with error feedback
```

**`Person` schema:**

| Field        | Type        |
|--------------|-------------|
| `name`       | `str`       |
| `age`        | `int`       |
| `city`       | `str`       |
| `occupation` | `str`       |
| `hobbies`    | `list[str]` |

**Retry behavior:**

| Error type          | Action                                      |
|---------------------|---------------------------------------------|
| `JSONDecodeError`   | Retry — tells model JSON was invalid        |
| `ValidationError`   | Retry — tells model which fields failed     |
| `APIStatusError`    | Retry — Groq API issue (no content feedback)|

---

# Day 4 — Multi-turn Conversational Loop

**Goals:** Build an interactive chat agent with context management — keep conversations going without blowing the token budget.

| File | Command |
|------|---------|
| Multi-turn chat | `python3 Day_4_multiturn_conversational_loop.py [strategy]` |

**Strategies** (pass as first argument; default is `token`):

| Strategy  | Class               | How memory works                                      |
|-----------|---------------------|-------------------------------------------------------|
| `token`   | `TokenManagedChat`  | Drop oldest messages when context exceeds token budget |
| `sliding` | `SlidingWindowChat` | Keep only the last N user/assistant turn pairs        |
| `summary` | `SummarizingChat`   | Compress old history into a summary when threshold hit |

```bash
python3 Day_4_multiturn_conversational_loop.py token
python3 Day_4_multiturn_conversational_loop.py sliding
python3 Day_4_multiturn_conversational_loop.py summary
```

**CLI commands during chat:** `exit` · `history` · `clear`

**Architecture:**

```
BaseChat          — shared Groq client, API retry (1s → 2s → 4s), turn storage
├── SlidingWindowChat
├── TokenManagedChat
└── SummarizingChat
ChatApp           — CLI loop; picks strategy via argv
```

**Key concepts:**

- `_save_turn()` — only stores user/assistant pairs after a successful API call
- `BaseChat.chat()` — abstract method (`NotImplementedError`); each strategy implements its own memory logic (polymorphism)
- Token budget uses `tiktoken` (`cl100k_base`) with a character-based fallback

---

# Day 5 — Chain-of-Thought (CoT) Prompting

**Goals:** Compare two reasoning prompt styles — natural step-by-step instructions vs structured XML tags — and analyze token usage and reasoning quality.

| File | Command |
|------|---------|
| CoT experiment | `python3 Day_5_chain_of_thought_(COT).py` |

```bash
# Default 6 math/logic questions
python3 Day_5_chain_of_thought_(COT).py

# Single custom question
python3 Day_5_chain_of_thought_(COT).py -q "What is 2+2?"

# Custom output file and token limit
python3 Day_5_chain_of_thought_(COT).py --max-tokens 500 -o my_results.json
```

| Flag | Description |
|------|-------------|
| `-q`, `--question` | Question to run (repeatable; uses defaults if omitted) |
| `--max-tokens` | Max completion tokens per API call (default: 1000) |
| `-o`, `--output` | JSON results path (default: `cot_results.json`) |

**Two prompting styles per question:**

| Style | Approach |
|-------|----------|
| Step-by-step | System prompt asks the model to solve problems step by step |
| XML tags | Model must use `<thinking>` for reasoning and `<answer>` for the final result |

**Classes:**

- `CoTExperiment` — runs both styles, API retry with backoff, per-question error handling, saves JSON
- `CoTAnalyzer` — extracts XML answers, counts reasoning steps, prints token/reasoning comparison report

**Output:** Console report plus JSON file with full answers and token counts per question.

---

# Day 8 — Tools as JSON Schemas

**Goals:** Define tools as JSON schemas, run the tool-use loop (`request → tool_use → tool_result → final response`), and execute Python handlers for each tool call.

| File | Command |
|------|---------|
| Tool schemas demo | `python3 Day_8_tools_as_Json_schemas.py` |

```bash
python3 Day_8_tools_as_Json_schemas.py
python3 Day_8_tools_as_Json_schemas.py --max-rounds 3
```

**Built-in tools:** `calculator`, `get_weather`, `search_web` (mock)

**Flow:**

```
User message → Groq picks tool → execute_tool() → role=tool result → Groq final answer
```

**Key concepts:**

- Tool schemas are passed in the `tools=` parameter to Groq
- Each tool call returns a string in a `role: "tool"` message
- Multi-round loop until the model responds with text only

---

# Day 9 — Tool-Use Agent Loop

**Goals:** Expand the agent with multiple real tools and a while-loop that keeps calling Groq until `finish_reason == "stop"`.

| File | Command |
|------|---------|
| Multi-tool loop | `python3 Day_9_tool_use_loop.py` |

```bash
python3 Day_9_tool_use_loop.py
```

**Tools:** `calculator`, `get_current_time`, `fetch_joke`

**Flow:**

```
while not done:
    response = Groq(messages + tools)
    if tool_calls → execute each → append tool results → continue
    if stop → return final answer
```

Edit the hard-coded `query` at the bottom of the script to test different prompts.

---

# Day 10 — Simultaneous Tool Use

**Goals:** Handle multiple tool calls in a single model response, run real web search, and study Groq `tool_choice` modes.

| File | Command |
|------|---------|
| Multi-tool + search | `python3 Day_10_tool_use_simulteniously.py` |

```bash
python3 Day_10_tool_use_simulteniously.py
python3 Day_10_tool_use_simulteniously.py --tool-choice required
python3 Day_10_tool_use_simulteniously.py --force-tool calculator
```

| Flag | Description |
|------|-------------|
| `-q`, `--question` | User question (default triggers calculator + time + search) |
| `--tool-choice` | `auto`, `none`, or `required` |
| `--force-tool` | Force a specific tool (`calculator`, `get_current_time`, `search_web`) |
| `--max-rounds` | Max tool rounds (default: 5) |

**Tools:** `calculator`, `get_current_time`, `search_web`

**Web search providers:**

| Provider | Key | Notes |
|----------|-----|-------|
| Serper | `SERPER_API_KEY` | Google results (optional) |
| DuckDuckGo | None | Free fallback |

**Class:** `ExecuteTool` — reusable tool executor used by later days.

---

# Day 11 — Context Q&A with Web Search Fallback

**Goals:** Load local sandbox files into context, answer from context when possible, and fall back to `search_web` only when needed.

| File | Command |
|------|---------|
| Context + search | `python3 Day_11_tool_use_read_write.py -q "Your question"` |

```bash
python3 Day_11_tool_use_read_write.py -q "How to learn and develop an AI agent?"
```

**Context files** (in `sandbox/`): `todo.txt`, `Health_benifits.md`

**Flow:**

1. Load file contents into the system prompt
2. Groq checks context → answer directly if found
3. If not found → call `search_web`
4. Return final answer using context or search results

| Flag | Description |
|------|-------------|
| `-q`, `--question` | Question to ask (required) |

---

# Day 12 — Tool Dispatcher & Structured Logging

**Goals:** Replace hard-coded tool routing with a generic dispatcher (registry dict → Python functions) and log every tool call to a structured JSON-lines file.

| File | Command |
|------|---------|
| Dispatcher + logging | `python3 day_12_logger_and_dispatcher.py -q "Your question"` |

```bash
python3 day_12_logger_and_dispatcher.py -q "How to learn and develop an AI agent?"
```

Same context-first Q&A flow as Day 11, with two additions:

**1. `ToolDispatcher` + `TOOL_REGISTRY`**

```python
TOOL_REGISTRY = {
    "search_web": _search_web_handler,
}
dispatcher.dispatch("search_web", {"query": "..."})
```

**2. `StructuredToolLogger`**

Writes one JSON object per line to `sandbox/tool_calls.log`:

```json
{
  "timestamp": "2026-07-18T12:34:56+00:00",
  "event": "tool_call",
  "tool": "search_web",
  "arguments": {"query": "..."},
  "result": "..."
}
```

| Flag | Description |
|------|-------------|
| `-q`, `--question` | Question to ask (required) |

**Shared module:** `all_tool_definition.py` — central tool JSON schemas for calculator, time, search, file read/write.

---

## Suggested Learning Path

```
Day 1  →  Setup & understand API parameters
         Day_1_test_setup.py
         Day_1_ten_run_with_diff_prams.py
         Day_1_main.py (optional — needs paid keys)

Day 2  →  Tokens, streaming, and output control
         Day_2_gork_token_counter.py
         Day_2_gorq_steaming.py
         Day_2_stop_sequence.py

Day 3  →  Structured data for agents
         Day_3_json_formatted_response.py
         Day_3_pedantic_validation_and_retry.py

Day 4  →  Multi-turn chat & memory management
         Day_4_multiturn_conversational_loop.py

Day 5  →  Chain-of-thought prompting
         Day_5_chain_of_thought_(COT).py

Day 8  →  Tools as JSON schemas
         Day_8_tools_as_Json_schemas.py

Day 9  →  Multi-tool agent loop
         Day_9_tool_use_loop.py

Day 10 →  Simultaneous tools + web search
         Day_10_tool_use_simulteniously.py

Day 11 →  Context files + search fallback
         Day_11_tool_use_read_write.py -q "Your question"

Day 12 →  Dispatcher registry + structured logging
         day_12_logger_and_dispatcher.py -q "Your question"
```

**Run all Groq scripts in order:**

```bash
python3 Day_1_test_setup.py
python3 Day_1_ten_run_with_diff_prams.py
python3 Day_2_gork_token_counter.py
python3 Day_2_gorq_steaming.py
python3 Day_2_stop_sequence.py
python3 Day_3_json_formatted_response.py
python3 Day_3_pedantic_validation_and_retry.py
python3 Day_4_multiturn_conversational_loop.py token    # interactive — type exit to quit
python3 Day_5_chain_of_thought_(COT).py
python3 Day_8_tools_as_Json_schemas.py
python3 Day_9_tool_use_loop.py
python3 Day_10_tool_use_simulteniously.py
python3 Day_11_tool_use_read_write.py -q "How to learn and develop an AI agent?"
python3 day_12_logger_and_dispatcher.py -q "How to learn and develop an AI agent?"
```

---

## Troubleshooting

### `python: command not found`

Use `python3` instead of `python`.

### `ModuleNotFoundError` (groq, tiktoken, pydantic)

```bash
source venv01/bin/activate
pip install -r requirements.txt
```

### `GROQ_API_KEY not found` / `GroqError: api_key must be set`

Add to `.env`:

```env
GORQ_API_KEY=gsk-your-key-here
```

### `model_decommissioned` / `mixtral-8x7b-32768`

Groq removed Mixtral. All scripts should use `llama-3.3-70b-versatile`.

### `Invalid JSON: Expecting value: line 1 column 1`

Model returned JSON wrapped in markdown fences. Day 3 scripts use `extract_json()` to handle this. If you see this in your own code, strip ` ```json ` before `json.loads()`.

### OpenAI / Anthropic quota errors

Paid credits required. Use Groq scripts for free learning.

### Gemini `429 RESOURCE_EXHAUSTED`

Free quota used up. Wait for reset or use Groq.

### Day 5 writes `cot_results.json`

Generated by default when running the CoT experiment. Add to `.gitignore` if you do not want to commit experiment output.

### `tool_use_failed` from Groq (Day 8–12)

Groq sometimes rejects malformed tool output. Day 10–12 retry with backoff; Day 11/12 also recover parsed tool calls from the error payload when possible. Re-run the script or simplify the question if retries fail.

### Web search returns no results (Day 10–12)

DuckDuckGo HTML parsing can be flaky. Set `SERPER_API_KEY` in `.env` for more reliable results, or try a different query.

### Day 12 writes `sandbox/tool_calls.log`

Append-only JSON-lines log of every tool dispatch. Safe to delete between runs; it is recreated automatically.

---

## Security Notes

- **Never commit `.env`** — listed in `.gitignore`
- **Never put API keys** in source code, `requirements.txt`, or README
- **Do not store personal info** in `.env`
- Revoke and regenerate keys if accidentally exposed

---

## Dependencies

Core packages from `requirements.txt`:

```
groq==1.5.0
requests==2.34.2
tiktoken==0.13.0
pydantic==2.13.4
python-dotenv==1.2.2
openai==2.43.0
anthropic==0.111.0
google-genai==2.10.0
tenacity==9.1.4
```

See `requirements.txt` for the full pinned list.
