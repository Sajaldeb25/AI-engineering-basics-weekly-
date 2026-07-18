"""
Day 11 — Context-first Q&A with web search fallback

Flow:
  1. Load todo.txt + Health_benifits.md into Groq context
  2. Groq checks context → answer directly if found
  3. If not found → Groq calls search_web
  4. Search results go back to Groq → final answer

Run:
    python3 Day_11_tool_use_read_write.py -q "How to learn and develop an AI agent?"
"""

import argparse
import json
import os
import re
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from groq import APIStatusError, Groq

from all_tool_definition import SEARCH_WEB_TOOL
from Day_10_tool_use_simulteniously import ExecuteTool

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_ROUNDS = 5
MAX_API_RETRIES = 3

TOOLS = [SEARCH_WEB_TOOL]
SOURCE_FILES = ["todo.txt", "Health_benifits.md"]

SANDBOX_DIR = os.path.join(os.getcwd(), "sandbox")
os.makedirs(SANDBOX_DIR, exist_ok=True)


def _safe_path(filename: str) -> str:
    basename = os.path.basename(filename)
    if not basename:
        raise ValueError("Invalid filename")
    return os.path.join(SANDBOX_DIR, basename)


def file_read(filename: str) -> str:
    safe_path = _safe_path(filename)
    if not os.path.exists(safe_path):
        return f"Error: File '{filename}' not found."
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as error:
        return f"Error reading file: {error}"


def get_groq_api_key():
    return os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")


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
    return payload


def parse_failed_tool_call(failed_generation: str) -> tuple[str, dict[str, Any]] | None:
    """Recover when Groq returns tool_use_failed with malformed tool output."""
    if not failed_generation:
        return None
    match = re.search(r"<function=(\w+)=?(\{.*\})</function>", failed_generation)
    if not match:
        return None
    try:
        return match.group(1), json.loads(match.group(2))
    except json.JSONDecodeError:
        return None


def _append_synthetic_tool_result(messages: list[dict], tools_called: list[str], name: str, args: dict[str, Any], result: str) -> None:

    tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
    tools_called.append(name)
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tool_call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result,
    })


def _load_source_context() -> str:
    parts = []
    for filename in SOURCE_FILES:
        content = file_read(filename)
        parts.append(f"--- {filename} ---\n{content}")

    print(f"Source context: {parts}")
    return "\n\n".join(parts)


def _build_system_prompt(source_context: str) -> str:
    source_list = ", ".join(SOURCE_FILES)
    return (
        "You answer user questions using the file context below.\n\n"
        f"Context files: {source_list}\n\n"
        "Rules:\n"
        "1. First search for the answer in the provided context.\n"
        "2. If the answer is clearly in the context, reply directly — do NOT call any tool.\n"
        "3. If the answer is NOT in the context, call the search_web tool with a focused query.\n"
        "4. After you receive search_web results, give the final answer to the user.\n"
        "5. Do not invent facts not supported by the context or search results.\n\n"
        f"File context:\n{source_context}"
    )


def run_agent(user_message: str, max_rounds: int = MAX_ROUNDS) -> dict:
    """
    Single Groq tool loop:
      question + file context + search_web tool → answer or web search → final answer
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise SystemExit("❌ GROQ_API_KEY (or GORQ_API_KEY) not found in .env")

    client = Groq(api_key=api_key)
    executor = ExecuteTool()
    tools_called: list[str] = []

    print(f"\n🧑 User: {user_message}\n")
    print(f"📁 Sandbox: {SANDBOX_DIR}\n")
    print(f"📄 Context files: {', '.join(SOURCE_FILES)}\n")

    source_context = _load_source_context()
    print("📖 Loaded file context for Groq.\n")

    messages = [
        {"role": "system", "content": _build_system_prompt(source_context)},
        {"role": "user", "content": user_message},
    ]

    def call_api() -> Any | None:
        wait = 1
        last_error = None
        for attempt in range(MAX_API_RETRIES):
            try:
                return client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=1000,
                )
            except APIStatusError as error:
                last_error = error
                body = getattr(error, "body", None) or {}
                error_info = body.get("error", {}) if isinstance(body, dict) else {}
                code = error_info.get("code") if isinstance(error_info, dict) else None
                failed_generation = (
                    error_info.get("failed_generation", "")
                    if isinstance(error_info, dict)
                    else ""
                )
                recovered = parse_failed_tool_call(failed_generation)
                if code == "tool_use_failed" and recovered:
                    name, args = recovered
                    print(f"⚠️  tool_use_failed — recovered {name}({args})")
                    result = executor.execute_tool(name, args)
                    preview = result[:120] + ("..." if len(result) > 120 else "")
                    print(f"   → {preview}")
                    _append_synthetic_tool_result(messages, tools_called, name, args, result)
                    return None
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
        print("=" * 72)
        print(f"ROUND {round_num}")
        print("=" * 72)

        response = call_api()
        if response is None:
            continue

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            final_answer = message.content or ""
            print(f"\n✅ FINAL: {final_answer}")
            break

        print(f"\nTool calls ({len(tool_calls)}):")
        messages.append(assistant_message_from_response(message))

        for tc in tool_calls:
            print(f"  • {tc.function.name}({tc.function.arguments})")
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as error:
                result = f"Error: invalid JSON arguments: {error}"
            else:
                result = executor.execute_tool(tc.function.name, args)
                tools_called.append(tc.function.name)
                preview = result[:120] + ("..." if len(result) > 120 else "")
                print(f"   → {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        print()
    else:
        final_answer = "Max rounds reached without final answer."

    return {
        "tools_called": tools_called,
        "web_search_used": "search_web" in tools_called,
        "tool_call_count": len(tools_called),
        "final_answer": final_answer,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Day 11 — context Q&A with web search fallback")
    parser.add_argument("-q", "--question", required=True, help="Question to ask")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = run_agent(args.question)
    print(f"\nTools used: {stats['tools_called'] or '(none)'}")
