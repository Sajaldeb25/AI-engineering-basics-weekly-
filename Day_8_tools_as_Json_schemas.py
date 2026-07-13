"""
Day 8 — Tools as JSON Schemas

Define tools the model may call, then trace one full turn by hand:

    request → tool_use block → tool_result → final response

Run:
    python3 Day_8_tools_as_Json_schemas.py           # print manual trace
    python3 Day_8_tools_as_Json_schemas.py --live    # optional live Groq demo
"""


""" 
---- Why we need tools ----

-Without tools:
You: “What’s the weather in Paris and what is 15 × 7?”
Model: “It’s probably around 18°C and sunny… 15 times 7 is 105.”
Maybe right. Maybe wrong. It has no real connection to weather or a calculator.

-With tools:

You: same question
Model: “I need get_weather(city=Paris) and calculator(15, 7, multiply).”
Your code: actually calls weather API → 18°C, sunny; runs math → 105
Model: “Paris is 18°C and sunny. 15 × 7 = 105.”

Now the answer uses real data and real computation, not guesses.
"""

import argparse
import json
from typing import Any

# ---------------------------------------------------------------------------
# 1. Three tools defined as JSON Schema (Groq / OpenAI-compatible format)
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
# 2. Stub implementations (code will run these — not the model)
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
    # Mock data — real app would call a weather API
    temp = 18 if units == "metric" else 64
    unit_label = "°C" if units == "metric" else "°F"
    return {
        "city": city,
        "temperature": temp,
        "unit": unit_label,
        "condition": "sunny",
    }


def search_web_impl(query: str, num_results: int = 3) -> list[dict]:
    # Mock data — real app would call a search API
    return [
        {"title": f"Result {i + 1} for '{query}'", "url": f"https://example.com/{i + 1}"}
        for i in range(num_results)
    ]


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call to the matching implementation."""
    if name == "calculator":
        return calculator_impl(arguments["a"], arguments["b"], arguments["operation"])
    if name == "get_weather":
        return get_weather_impl(arguments["city"], arguments.get("units", "metric"))
    if name == "search_web":
        return search_web_impl(arguments["query"], arguments.get("num_results", 3))
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# 3. Manual trace — one full tool-use turn (trace this on paper)
# ---------------------------------------------------------------------------
#
# Scenario: user asks for weather in Paris AND 15 × 7.
# Model picks calculator first (could also call get_weather in parallel).
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 1 — REQUEST (you → API)                                            │
# │   messages: [user question]                                             │
# │   tools:    TOOLS (schemas above)                                         │
# └─────────────────────────────────────────────────────────────────────────┘
#                                    ↓
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 2 — ASSISTANT tool_use block (API → you)                           │
# │   role: assistant                                                         │
# │   tool_calls: [{ id, type:"function", function:{ name, arguments } }]   │
# │   finish_reason: "tool_calls"                                            │
# └─────────────────────────────────────────────────────────────────────────┘
#                                    ↓
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 3 — tool_result (you → API)                                        │
# │   role: tool                                                              │
# │   tool_call_id: matches the id from step 2                                │
# │   content: JSON string or plain text from execute_tool()                  │
# └─────────────────────────────────────────────────────────────────────────┘
#                                    ↓
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ STEP 4 — FINAL RESPONSE (API → you)                                     │
# │   Second request sends full history: user + assistant(tool_calls) + tool │
# │   Assistant returns natural-language answer using tool result.            │
# └─────────────────────────────────────────────────────────────────────────┘

MANUAL_TRACE = {
    "step_1_request": {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Paris, and what is 15 times 7?",
            }
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
    },
    "step_2_assistant_tool_use": {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_weather_001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Paris", "units": "metric"}),
                },
            },
            {
                "id": "call_calc_002",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"a": 15, "b": 7, "operation": "multiply"}),
                },
            },
        ],
        "finish_reason": "tool_calls",
    },
    "step_3_tool_results": [
        {
            "role": "tool",
            "tool_call_id": "call_weather_001",
            "content": json.dumps(
                {"city": "Paris", "temperature": 18, "unit": "°C", "condition": "sunny"}
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_calc_002",
            "content": "105",
        },
    ],
    "step_4_follow_up_request": {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Paris, and what is 15 times 7?",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris", "units": "metric"}),
                        },
                    },
                    {
                        "id": "call_calc_002",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"a": 15, "b": 7, "operation": "multiply"}),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_001",
                "content": json.dumps(
                    {"city": "Paris", "temperature": 18, "unit": "°C", "condition": "sunny"}
                ),
            },
            {
                "role": "tool",
                "tool_call_id": "call_calc_002",
                "content": "105",
            },
        ],
        "tools": TOOLS,
    },
    "step_4_final_response": {
        "role": "assistant",
        "content": (
            "In Paris, it's currently 18°C and sunny. "
            "Also, 15 times 7 equals 105."
        ),
        "finish_reason": "stop",
    },
}


def _pretty(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def print_manual_trace():
    """Print the full request → tool_use → tool_result → response flow."""
    print("=" * 72)
    print("DAY 8 — MANUAL TOOL-USE TRACE (trace this on paper)")
    print("=" * 72)

    print("\n📤 STEP 1 — REQUEST (you send to API)")
    print(_pretty(MANUAL_TRACE["step_1_request"]))

    print("\n📥 STEP 2 — ASSISTANT tool_use block (API returns)")
    print(_pretty(MANUAL_TRACE["step_2_assistant_tool_use"]))

    print("\n🔧 STEP 3 — tool_result (you execute tools, send back)")
    for result in MANUAL_TRACE["step_3_tool_results"]:
        print(_pretty(result))

    print("\n📤 STEP 4a — FOLLOW-UP REQUEST (full message history)")
    print(_pretty(MANUAL_TRACE["step_4_follow_up_request"]))

    print("\n📥 STEP 4b — FINAL RESPONSE (assistant answers in plain text)")
    print(_pretty(MANUAL_TRACE["step_4_final_response"]))

    print("\n" + "-" * 72)
    print("KEY RULES")
    print("-" * 72)
    print("• tool_call_id in tool_result MUST match id from tool_calls")
    print("• arguments is a JSON string — parse before calling your function")
    print("• You run execute_tool(); the model only picks name + arguments")
    print("• Second API call includes user + assistant(tool_calls) + tool messages")


def run_tool_loop_from_assistant(assistant_message: dict) -> list[dict]:
    """
    Pseudocode from step 2 → step 3: execute each tool_call and build tool results.
    """
    results = []
    for tool_call in assistant_message.get("tool_calls", []):
        if tool_call["type"] != "function":
            continue

        name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        output = execute_tool(name, arguments)

        results.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(output) if isinstance(output, (dict, list)) else str(output),
        })

    return results


def run_live_demo():
    """Optional: run a real tool-use turn against Groq."""
    import os

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")
    if not api_key:
        raise SystemExit("❌ GROQ_API_KEY (or GORQ_API_KEY) not found in .env")

    client = Groq(api_key=api_key)
    user_message = "What's the weather in Paris, and what is 15 times 7?"

    print("\n🌐 LIVE DEMO — calling Groq with tools\n")
    print(f"User: {user_message}\n")

    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_msg = response.choices[0].message
    tool_calls = assistant_msg.tool_calls or []

    print("STEP 2 — Assistant tool_calls:")
    for tc in tool_calls:
        print(f"  • {tc.function.name}({tc.function.arguments})")

    messages.append({
        "role": "assistant",
        "content": assistant_msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    })

    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        output = execute_tool(tc.function.name, args)
        content = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        print(f"STEP 3 — tool_result [{tc.id}]: {content}")

    final = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS,
    )

    print(f"\nSTEP 4 — Final: {final.choices[0].message.content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 8 — tools as JSON schemas")
    parser.add_argument("--live", action="store_true", help="Run live Groq tool-use demo")
    args = parser.parse_args()

    print_manual_trace()

    # Verify step 2 → step 3 with our stub implementations
    print("\n" + "=" * 72)
    print("EXECUTING STEP 2 → STEP 3 (stub implementations)")
    print("=" * 72)
    computed = run_tool_loop_from_assistant(MANUAL_TRACE["step_2_assistant_tool_use"])
    for item in computed:
        print(_pretty(item))

    if args.live:
        run_live_demo()
