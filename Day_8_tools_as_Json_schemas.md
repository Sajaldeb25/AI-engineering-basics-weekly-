# Day 8 — Tools as JSON Schemas

Guide for `Day_8_tools_as_Json_schemas.py`.

**Goal:** Teach the model to *request* helper functions (tools), while **your Python code** actually runs them, then feed results back so the model can answer with real data instead of guesses.

**Provider:** Groq · **Model:** `llama-3.3-70b-versatile`

---

## Why use tools?

| Without tools | With tools |
|---------------|------------|
| Model *guesses* weather, math, search | Model picks a tool + arguments |
| Answers may be wrong or outdated | Your code runs real (or mock) logic |
| No link to APIs or calculators | Model writes English from real results |

**One sentence:** The model is the planner; your code is the worker.

```
You ask → Model picks tools → YOUR code runs tools → Model answers
```

---

## How to run

```bash
# Activate venv and ensure .env has GORQ_API_KEY or GROQ_API_KEY
source venv01/bin/activate
python3 Day_8_tools_as_Json_schemas.py
python3 Day_8_tools_as_Json_schemas.py --max-rounds 5
```

| Flag | Meaning |
|------|---------|
| `--max-rounds N` | Stop after N tool rounds if the model never returns a final text answer (default: 5) |

---

## The four-step tool-use turn

```
STEP 1 — REQUEST
  You send: user message + tool schemas (TOOLS)

STEP 2 — TOOL USE
  Model returns: tool_calls (name + arguments)
  content is often None
  finish_reason: tool_calls

STEP 3 — TOOL RESULT
  Your code runs execute_tool()
  You send back: role=tool messages (tool_call_id must match)

STEP 4 — FINAL RESPONSE
  Second (or Nth) API call with full history
  Model returns plain text answer
```

Models may need **multiple rounds** (call weather first, then calculator later). This script loops until there are no more `tool_calls` or `max_rounds` is hit.

---

## The three tools

Defined in `TOOLS` as Groq/OpenAI-compatible JSON schemas.

| Tool | Parameters | Purpose |
|------|------------|---------|
| `calculator` | `a`, `b`, `operation` (`add` / `subtract` / `multiply` / `divide`) | Basic arithmetic |
| `get_weather` | `city` (required), `units` (`metric` / `imperial`) | Weather lookup (mocked) |
| `search_web` | `query` (required), `num_results` | Web search (mocked) |

Schemas are the **menu**. Implementations are the **kitchen**.

---

## File structure

| Section | What it does |
|---------|----------------|
| `TOOLS` | JSON schemas sent to the API |
| `calculator_impl` / `get_weather_impl` / `search_web_impl` | Stub functions your code runs |
| `execute_tool()` | Dispatches tool name → correct impl |
| `run_live_demo()` | Multi-round Groq agent loop |
| `__main__` | Parses `--max-rounds` and starts the demo |

### Who does what?

| Actor | Responsibility |
|-------|----------------|
| Groq model | Chooses tool name + JSON arguments |
| Your Python | Parses arguments, runs stubs, appends `tool` messages |
| You | Ask the question via `user_message` |

---

## Multi-round loop (important)

```text
messages = [system, user]

round 1:
  API → tool_calls: get_weather, calculator, search_web
  you → execute tools, append results to messages

round 2:
  API → final English answer (no tool_calls)
  done
```

If you only do **one** tool round, Step 4 may print `None` because the model asked for more tools and returned empty `content`.

---

## Key rules

1. `arguments` from the API is a **JSON string** — always `json.loads(...)` before calling your function.
2. Every `tool` message needs `tool_call_id` matching the assistant `tool_calls[].id`.
3. You must append the assistant message that contains `tool_calls` **before** appending tool results.
4. Keep looping while `tool_calls` is non-empty.
5. Stubs are mock data; replace them with real APIs when you build a production agent.

---

## Example demo question

```text
What's the weather in Paris, and what is 15 times 7?
Also, search the web for 'python programming'.
```

Typical successful output:

```text
ROUND 1
  Tool calls: get_weather, calculator, search_web
  → tool results: 18°C sunny, 105, fake search links

ROUND 2
  ✅ FINAL RESPONSE: weather + math + search summarized in English
```

---

## Related concepts (next steps)

- Replace stub weather/search with real HTTP APIs
- Add validation (Pydantic) on tool arguments before calling impls
- Persist conversation history across user turns (like Day 4 memory)
- Restrict tools with `tool_choice` (`auto` / `required` / named tool)

---

## Quick map to the code

| Idea | Code |
|------|------|
| Menu of tools | `TOOLS` |
| Worker functions | `*_impl` + `execute_tool` |
| Agent loop | `run_live_demo` → `for round_num in ...` |
| Retry flaky Groq tool format | `call_with_tools` + `tool_use_failed` |
| Entry | `if __name__ == "__main__"` |
