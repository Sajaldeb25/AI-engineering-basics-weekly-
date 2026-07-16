""" 
add tools to the agent
1. calculator
2. get_current_time
3. get_weather
4. convert_units
5. fetch_joke
6. word_count
"""
import os
import json
import math
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from groq import APIStatusError, Groq

load_dotenv()

MAX_API_RETRIES = 3

# -------------------------------
# 1. Tool definition
# -------------------------------
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, %, parentheses.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate, e.g. '4892 * 0.17 - 33'"
                }
            },
            "required": ["expression"]
        }
    }
}

get_current_time_tool = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time. Use for questions about what time it is now.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "Timezone to use. Examples: 'local', 'UTC', "
                        "'Asia/Dhaka', 'Asia/Kolkata', 'Asia/Kuala_Lumpur', "
                        "'Europe/Oslo', 'Europe/Berlin'."
                    ),
                }
            },
            "required": [],
        },
    },
}

# list of tools to use
tools = [calculator_tool, get_current_time_tool]

# -------------------------------
# 2. Tool execution function
# -------------------------------
def execute_calculator(expression: str) -> float:
    """
    Safely evaluate an arithmetic expression.
    Args:
        expression: The arithmetic expression to evaluate, e.g. '4892 * 0.17 - 33 + 100'
    Returns:
        The result of the evaluation as a float.
    Raises:
        ValueError: If the expression is invalid.
    
    """
    print(f"\n\n Evaluating expression: {expression}\n")
    safe_dict = {
        'abs': abs, 
        'round': round, 
        'pow': pow,
        'sqrt': math.sqrt,
        'pi': math.pi, 
        'e': math.e,
        'sin': math.sin, 
        'cos': math.cos, 
        'tan': math.tan,
        'log': math.log, 
        'exp': math.exp,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression '{expression}': {e}")


def execute_get_current_time(tz: str = "local") -> str:
    """Return current date/time as a readable string."""
    tz_name = (tz or "local").strip()
    key = tz_name.lower()

    try:
        if key == "local":
            now = datetime.now().astimezone()
            label = "local"
        elif key == "utc":
            now = datetime.now(timezone.utc)
            label = "UTC"
        else:
            now = datetime.now(ZoneInfo(tz_name))
            label = tz_name
    except ZoneInfoNotFoundError:
        return (
            f"Error: unknown timezone '{tz_name}'. "
            "Use IANA names like Asia/Kolkata, Asia/Dhaka, Europe/Oslo."
        )

    return f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')} ({label})"


# -------------------------------
# 3. Main tool‑use loop
# -------------------------------
def run_agent_with_tools(user_message: str):
    print(f"\n\n Running agent with tools...\n")
    print(f"User message: {user_message}\n")
    print(f"Tools: {tools}\n")
    # print(f"Max iterations: {max_iterations}\n")
    # print(f"Model: {model}\n") 


    client = Groq(api_key=os.getenv("GORQ_API_KEY"))
    model = "llama-3.3-70b-versatile"

    # Start conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with tools. "
                "Use calculator for math (percentages as decimals, e.g. 18% → 0.18 * x). "
                "Use get_current_time for questions about the current date or time."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    max_iterations = 5  # avoid infinite loops
    iteration = 0

    def call_with_tools():
        """Call Groq; retry on intermittent tool_use_failed errors."""
        wait = 1
        last_error = None
        for attempt in range(MAX_API_RETRIES):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=1000,
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

    while iteration < max_iterations:
        iteration += 1

        # Step 1: Call the model with tools (retry on tool_use_failed)
        response = call_with_tools()

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        print(f"\n\n Finish reason: {finish_reason}\n")
        
        # Step 2: Check if the model wants to call a tool
        if finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls
            # Append the assistant's message with tool_calls
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in tool_calls
                ]
            })  # includes tool_calls

            # Step 3: Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "calculator":
                    expression = function_args.get("expression")
                    try:
                        result = execute_calculator(expression)
                        result_str = str(result)
                    except ValueError as e:
                        result_str = f"Error: {e}"
                elif function_name == "get_current_time":
                    tz = function_args.get("timezone", "local")
                    result_str = execute_get_current_time(tz)
                    print(f"\n Current time ({tz}): {result_str}\n")
                else:
                    result_str = f"Unknown tool: {function_name}"

                # Step 4: Append tool response as a message with role 'tool'
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })

            # Loop will go back to the API call with the updated messages
            continue

        elif finish_reason == "stop":
            # Final answer
            return choice.message.content

        else:
            # Other finish reasons (e.g., length) – treat as error
            return f"Unexpected finish_reason: {finish_reason}"

    return "Max iterations reached without final answer."

# -------------------------------
# 4. Test with the given query
# -------------------------------
if __name__ == "__main__":
    query = "What time is it now in oslo, and what is 50% of 4892 minus 11 plus 444, then devide whole things by 12?"
    print(f"🧑 User: {query}")
    answer = run_agent_with_tools(query)
    print(f"🤖 Assistant: {answer}")