import os
import json
from dotenv import load_dotenv
from groq import Groq
import math

load_dotenv()

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

tools = [calculator_tool]

# -------------------------------
# 2. Tool execution function
# -------------------------------
def execute_calculator(expression: str) -> float:
    """Safely evaluate an arithmetic expression."""
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

# -------------------------------
# 3. Main tool‑use loop
# -------------------------------
def run_agent_with_tools(user_message: str):
    client = Groq(api_key=os.getenv("GORQ_API_KEY"))
    model = "llama-3.3-70b-versatile"

    # Start conversation
    messages = [{"role": "user", "content": user_message}]

    max_iterations = 5  # avoid infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Step 1: Call the model with tools
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",       # let model decide
            temperature=0.3,
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        # Step 2: Check if the model wants to call a tool
        if finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls
            # Append the assistant's message with tool_calls
            messages.append(choice.message.model_dump())  # includes tool_calls

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
    query = "What is 17% of 4892 minus 33?"
    print(f"🧑 User: {query}")
    answer = run_agent_with_tools(query)
    print(f"🤖 Assistant: {answer}")