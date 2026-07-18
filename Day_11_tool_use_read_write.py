"""
Day 11 — File read/write tools (sandboxed)

1. Add file_read and file_write — let the LLM interact with the filesystem safely
2. Observe: what happens when you give the LLM a tool it doesn't need? Does it use it anyway?

Run:
    python3 Day_11_tool_use_read_write.py                    # both scenarios
    python3 Day_11_tool_use_read_write.py --scenario file    # needs file tools
    python3 Day_11_tool_use_read_write.py --scenario no-file # observe unnecessary use
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

SANDBOX_DIR = os.path.join(os.getcwd(), "sandbox") # get current working directory and join with sandbox directory
os.makedirs(SANDBOX_DIR, exist_ok=True) # create sandbox directory if it doesn't exist


def _safe_path(filename: str) -> str:
    """Resolve the full path inside the sandbox and prevent directory traversal."""
    basename = os.path.basename(filename)
    if not basename:
        raise ValueError("Invalid filename")
    return os.path.join(SANDBOX_DIR, basename)


def file_read(filename: str) -> str:
    """Read the content of a file from the sandbox."""
    safe_path = _safe_path(filename)
    if not os.path.exists(safe_path):
        return f"Error: File '{filename}' not found."
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as error:
        return f"Error reading file: {error}"


def file_write(filename: str, content: str) -> str:
    """Write content to a file in the sandbox (overwrites if exists)."""
    safe_path = _safe_path(filename)
    try:

        print(f"\nWriting to {safe_path}")
        print(f"\nContent: {content}\n\n")

        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to '{filename}'."
    except OSError as error:
        return f"Error writing file: {error}"


FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the content of a file from the sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read (e.g. 'notes.txt')",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": (
                "Write content to a file in the sandbox directory. Overwrites if exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to write (e.g. 'notes.txt')",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write to the file",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in FILE_TOOLS}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "file_read":
        return file_read(arguments.get("filename", ""))
    
    if name == "file_write":
        return file_write(arguments.get("filename", ""), arguments.get("content", ""))
    
    return f"Error: unknown tool '{name}'"


def get_groq_api_key():
    return os.getenv("GORQ_API_KEY")


def assistant_message_from_response(message) -> dict:
    tool_calls = message.tool_calls or []
    payload = {"role": "assistant", "content": message.content}
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]

    print(f"\n Prepared payload for API calls, with tool function names and arguments: {payload}\n")
    print(f"--------------------------------\n")
    return payload


def run_agent(user_message: str, max_rounds: int = MAX_ROUNDS) -> dict:
    """
    Run the file-tool agent loop. Returns stats for observation:
      - tools_called: list of tool names invoked
      - file_tools_used: bool
      - final_answer: str
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise SystemExit("❌ GORQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)
    tools_called: list[str] = []

    print(f"\n🧑 User: {user_message}\n")
    print(f"📁 Sandbox: {SANDBOX_DIR}\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with sandbox file tools: file_read, file_write. "
                "Only use file tools when the user explicitly asks to read or write a file. "
                "For general questions (math, explanations, chat), answer directly without tools."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    print(f"\n Prepared messages for API calls: {messages}\n")

    def call_api():
        wait = 1
        last_error = None
        for attempt in range(MAX_API_RETRIES):
            try:
                return client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=FILE_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
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

    final_answer = ""

    for round_num in range(1, max_rounds + 1):
        print(f"\n---------------  Start of round {round_num} ---------------\n")

        print("=" * 72)
        print(f"ROUND {round_num}")
        print("=" * 72)

        response = call_api()
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            final_answer = message.content or ""
            print(f"\n✅ FINAL: {final_answer}")
            break

        print(f"Tool calls ({len(tool_calls)}):")
        messages.append(assistant_message_from_response(message))

        for tc in tool_calls:
            print(f"  • {tc.function.name}({tc.function.arguments})")
            tools_called.append(tc.function.name)

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as error:
                result = f"Error: invalid JSON arguments: {error}"
            else:
                print(f"\n Executing tool: {tc.function.name} with arguments: {args}\n")
                result = execute_tool(tc.function.name, args)
                preview = result[:120] + ("..." if len(result) > 120 else "")
                print(f"Preview 120 characters:    → {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        print(f"\n Prepared messages for API calls: {messages}\n")

        print(f"---------------  End of round {round_num} ---------------")

        print()
    else:
        final_answer = "Max rounds reached without final answer."

    stats = {
        "tools_called": tools_called,
        "file_tools_used": len(tools_called) > 0,
        "tool_call_count": len(tools_called),
        "final_answer": final_answer,
    }
    return stats


def print_observation(label: str, stats: dict, needs_file_tools: bool):
    print("\n" + "-" * 72)
    print(f"📊 OBSERVATION — {label}")
    print("-" * 72)
    print(f"  File tools needed for this task?  {needs_file_tools}")
    print(f"  File tools actually called?       {stats['file_tools_used']}")
    print(f"  Tools invoked:                   {stats['tools_called'] or '(none)'}")
    print(f"  Total tool calls:                {stats['tool_call_count']}")

    if needs_file_tools and not stats["file_tools_used"]:
        print("  ⚠️  Expected file tools but none were used.")
    elif not needs_file_tools and stats["file_tools_used"]:
        print("  🔍 Model used file tools even though the question did not require them.")
    elif not needs_file_tools and not stats["file_tools_used"]:
        print("  ✅ Model answered without unnecessary file tool use.")
    else:
        print("  ✅ Model used file tools as expected.")


SCENARIOS = {
    "file": {
        "query": (
            "Write a file called 'todo.txt' in the sandbox with three tasks for learning "
            "Python APIs. Then read the file back and summarize the tasks."
        ),
        "needs_file_tools": True,
    },
    "no-file": {
        "query": "What is 15 times 7? Explain the answer briefly in one sentence.",
        "needs_file_tools": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Day 11 — sandbox file read/write tools")
    parser.add_argument(
        "--scenario",
        choices=["file", "no-file", "both"],
        default="both",
        help="Which demo to run (default: both)",
    )
    parser.add_argument(
        "-q", "--question",
        help="Custom question (overrides scenario query)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n Arguments Scenario: {args.scenario}\n")
    print(f"\n Arguments Question: {args.question}\n")

    if args.question:
        stats = run_agent(args.question)
        print_observation("custom query", stats, needs_file_tools=False)
    elif args.scenario == "both":
        for name in ("file", "no-file"):
            cfg = SCENARIOS[name]
            print("\n" + "#" * 72)
            print(f"SCENARIO: {name}")
            print("#" * 72)
            stats = run_agent(cfg["query"])
            print_observation(name, stats, cfg["needs_file_tools"])
    elif args.only_read:
        stats = run_agent(args.question)
        print_observation("custom query", stats, needs_file_tools=True)
    elif args.only_write:
        stats = run_agent(args.question)
        print_observation("custom query", stats, needs_file_tools=True)
    else:
        cfg = SCENARIOS[args.scenario]
        stats = run_agent(cfg["query"])
        print_observation(args.scenario, stats, cfg["needs_file_tools"])
