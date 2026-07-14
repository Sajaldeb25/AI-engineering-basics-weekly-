"""
Day 8 — Tools as JSON Schemas

Define tools the model may call, then run a tool-use turn:

    request → tool_use → tool_result → final response

Why tools?
  Without tools, the model guesses (weather, math, search).
  With tools, the model picks a function; YOUR code runs it; the model
  answers using real results.

Run:
    python3 Day_8_tools_as_Json_schemas.py
"""

import argparse
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import APIStatusError, Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_ROUNDS = 5
MAX_API_RETRIES = 3

# ---------------------------------------------------------------------------
# 1. Tool schemas (menu the model can choose from)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs basic arithmetic (+, -, *, /) on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Arithmetic operation to perform",
                    },
                },
                "required": ["a", "b", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g. 'Paris')",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "description": "Temperature units (default: metric)",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a query and return top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 3)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 2. Stub implementations (YOUR code runs these — not the model)
# ---------------------------------------------------------------------------

def calculator_impl(a: float, b: float, operation: str) -> float:
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")
    if operation == "divide" and b == 0:
        raise ValueError("Division by zero")
    return ops[operation](a, b)


def get_weather_impl(city: str, units: str = "metric") -> dict:
    # Mock data — a real app would call a weather API
    temp = 18 if units == "metric" else 64
    unit_label = "°C" if units == "metric" else "°F"
    return {
        "city": city,
        "temperature": temp,
        "unit": unit_label,
        "condition": "sunny",
    }


def search_web_impl(query: str, num_results: int = 3) -> list[dict]:
    # Mock data — a real app would call a search API
    return [
        {"title": f"Result {i + 1} for '{query}'", "url": f"https://example.com/{i + 1}"}
        for i in range(num_results)
    ]


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Run the matching stub function for a tool call."""
    if name == "calculator":
        return calculator_impl(arguments["a"], arguments["b"], arguments["operation"])
    if name == "get_weather":
        return get_weather_impl(arguments["city"], arguments.get("units", "metric"))
    if name == "search_web":
        return search_web_impl(arguments["query"], arguments.get("num_results", 3))
    raise ValueError(f"Unknown tool: {name}")


def get_groq_api_key():
    return os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")


# ---------------------------------------------------------------------------
# 3. Live multi-round tool-use loop
# ---------------------------------------------------------------------------

def run_live_demo(max_rounds: int = MAX_ROUNDS):
    """
    Call Groq with tools. If the model returns tool_calls, execute them,
    append results, and call again until it returns a final text answer.
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise SystemExit("❌ GROQ_API_KEY (or GORQ_API_KEY) not found in .env")

    client = Groq(api_key=api_key)
    user_message = (
        "What's the weather in Paris, and what is 15 times 7? "
        "Also, search the web for 'python programming'."
    )

    print("\n🌐 LIVE DEMO — Groq tools (multi-round loop)\n")
    print(f"User: {user_message}\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use tools for weather, arithmetic, and web search. "
                "Call every tool you need before giving a final answer."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    def call_with_tools():
        wait = 1
        last_error = None


        # call api Groq API call. with max limit 

        for attempt in range(MAX_API_RETRIES):
            try:
                return client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except APIStatusError as error:
                last_error = error
                body = getattr(error, "body", None) or {}
                code = (
                    body.get("error", {}).get("code")
                    if isinstance(body, dict)
                    else None
                )
                if code == "tool_use_failed" and attempt < MAX_API_RETRIES - 1:
                    print(f"⚠️  tool_use_failed, retrying in {wait}s...")
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("Groq call failed")

    for round_num in range(1, max_rounds + 1):
        print("=" * 72)
        print(f"ROUND==> {round_num}")
        print("=" * 72)

        response = call_with_tools()
        assistant_msg = response.choices[0].message
        tool_calls = assistant_msg.tool_calls or []

        print(f"Assistant content: {assistant_msg.content}")

        if not tool_calls:
            print(f"\n✅ FINAL RESPONSE (round {round_num}):")
            print(assistant_msg.content or "(empty content)")
            return

        print(f"Tool calls ({len(tool_calls)}):")
        for tc in tool_calls:
            print(f"  • {tc.function.name}({tc.function.arguments})")

        messages.append({
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            output = execute_tool(tc.function.name, args)
            content = (
                json.dumps(output) if isinstance(output, (dict, list)) else str(output)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })
            print(f"  → tool_result [{tc.id}]: {content}")

        print()

    print(f"\n⚠️  Stopped after {max_rounds} rounds without a final text answer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 8 — tools as JSON schemas")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_ROUNDS,
        help=f"Max tool rounds before stopping (default: {MAX_ROUNDS})",
    )
    args = parser.parse_args()
    run_live_demo(max_rounds=args.max_rounds)
