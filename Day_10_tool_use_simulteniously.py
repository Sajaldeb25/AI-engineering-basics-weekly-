"""
Day 10 — Simultaneous tool use

1. Multi-tool calls in one response — handle when the LLM calls 2+ tools at once
2. Real web search via DuckDuckGo (free, no key) or Serper (optional SERPER_API_KEY)
3. Study tool choice: auto vs none vs required vs force a specific tool

Run:
    python3 Day_10_tool_use_simulteniously.py
    python3 Day_10_tool_use_simulteniously.py --tool-choice auto
    python3 Day_10_tool_use_simulteniously.py --tool-choice required
    python3 Day_10_tool_use_simulteniously.py --force-tool calculator
"""

import argparse
import json
import math
import os
import re
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dotenv import load_dotenv
from groq import APIStatusError, Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_ROUNDS = 5
MAX_API_RETRIES = 3

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression. Use decimals for percentages (18% → 0.18 * x).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression to evaluate, e.g. '4892 * 0.17 - 33'",
                }
            },
            "required": ["expression"],
        },
    },
}

GET_CURRENT_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get current date and time for a timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone or 'local'/'UTC'. "
                        "Examples: Asia/Dhaka, Asia/Kolkata, Europe/Oslo."
                    ),
                }
            },
            "required": [],
        },
    },
}

SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information, news, or facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {
                    "type": "string",
                    "description": "Number of results as digits, e.g. '3' or '5' (default '3', max '5')",
                },
            },
            "required": ["query"],
        },
    },
}

TOOLS = [CALCULATOR_TOOL, GET_CURRENT_TIME_TOOL, SEARCH_WEB_TOOL]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}



class ExecuteTool:
    
    def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and always return a string (required for role=tool)."""
        if name == "calculator":
            return self.execute_calculator(arguments.get("expression", ""))
        if name == "get_current_time":
            return self.execute_get_current_time(arguments.get("timezone", "local"))
        if name == "search_web":
            raw_num = arguments.get("num_results", 3)
            try:
                num_results = int(raw_num)
            except (TypeError, ValueError):
                num_results = 3
            return self.execute_search_web(arguments.get("query", ""), num_results)
        return f"Error: unknown tool '{name}'"

    def execute_calculator(self, expression: str) -> str:
        safe_dict = {
            "abs": abs,
            "round": round,
            "pow": pow,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
        }
        try:
            result = float(eval(expression, {"__builtins__": {}}, safe_dict))
            return str(result)
        except Exception as error:
            return f"Error: invalid expression '{expression}': {error}"


    def execute_get_current_time(self, tz: str = "local") -> str:
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
            return f"Error: unknown timezone '{tz_name}'"
        return f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')} ({label})"


    def search_serper(self, query: str, num_results: int) -> list[dict]:
        print(f"\nSearching Serper for {query} with {num_results} results")
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return []

        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        print(f"\nResults from Serper: {results}")
        return results


    def search_duckduckgo(self, query: str, num_results: int) -> list[dict]:
        print(f"Searching DuckDuckGo for {query} with {num_results} results")
        """Free web search via DuckDuckGo HTML (no API key)."""
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Day10-learning-bot)"},
            timeout=10,
        )
        response.raise_for_status()
        html = response.text

        # Parse result blocks: <a class="result__a" href="...">title</a>
        links = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            re.IGNORECASE,
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div|span)>',
            html,
            re.IGNORECASE | re.DOTALL,
        )

        results = []
        for i, (url, title) in enumerate(links[:num_results]):
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            clean_url = urllib.parse.unquote(url)
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
            results.append({"title": clean_title, "url": clean_url, "snippet": snippet})

        if not results:
            # Fallback: DuckDuckGo instant answer API (sparse but keyless)
            params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
            r = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", ""),
                })
            for topic in data.get("RelatedTopics", [])[: num_results - 1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic["Text"][:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic["Text"],
                    })

        return results[:num_results]


    def execute_search_web(self, query: str, num_results: int = 3) -> str:
        num_results = max(1, min(num_results, 5))
        provider = "Serper" if os.getenv("SERPER_API_KEY") else "DuckDuckGo"
        try:
            if provider == "Serper":
                results = self.search_serper(query, num_results)
                if not results:
                    results = self.search_duckduckgo(query, num_results)
                    provider = "DuckDuckGo (Serper returned nothing)"
            else:
                results = self.search_duckduckgo(query, num_results)

            if not results:
                return f"No results found for '{query}'."

            lines = [f"Search provider: {provider}"]
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item['title']}")
                if item.get("url"):
                    lines.append(f"   {item['url']}")
                if item.get("snippet"):
                    lines.append(f"   {item['snippet']}")
            return "\n".join(lines)
        except requests.RequestException as error:
            return f"Error: web search failed: {error}"




# ---------------------------------------------------------------------------
# Tool choice helpers (study auto / none / required / specific)
# ---------------------------------------------------------------------------

def build_tool_choice(mode: str, force_tool: str | None):
    """
    Groq tool_choice options:
      auto     — model decides whether to call tools
      none     — never call tools (text only)
      required — must call at least one tool
      force    — must call a specific tool (--force-tool NAME)
    """
    if force_tool:
        if force_tool not in TOOL_NAMES:
            raise SystemExit(
                f"Unknown tool '{force_tool}'. Choose from: {', '.join(sorted(TOOL_NAMES))}"
            )
        return {"type": "function", "function": {"name": force_tool}}

    if mode == "auto":
        return "auto"
    if mode == "none":
        return "none"
    if mode == "required":
        return "required"
    raise SystemExit(f"Unknown tool-choice mode: {mode}")


def get_groq_api_key():
    return os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")


def assistant_message_from_response(message) -> dict:
    """Build a Groq-safe assistant message (no extra fields from model_dump)."""
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


def run_agent(
    user_message: str,
    tool_choice_mode: str = "auto",
    force_tool: str | None = None,
    max_rounds: int = MAX_ROUNDS,
):
    api_key = get_groq_api_key()
    if not api_key:
        raise SystemExit("❌ GROQ_API_KEY (or GORQ_API_KEY) not found in .env")

    client = Groq(api_key=api_key)
    tool_choice = build_tool_choice(tool_choice_mode, force_tool)

    print(f"\n🔧 tool_choice = {tool_choice!r}\n")
    print(f"🧑 User: {user_message}\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with tools: calculator, get_current_time, "
                "search_web. Use tools for math, timezones, and web facts. "
                "You may call multiple tools in one turn when needed."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    def call_api():
        wait = 1
        last_error = None
        for attempt in range(MAX_API_RETRIES):
            try:
                return client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=tool_choice,
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

    for round_num in range(1, max_rounds + 1):
        print("=" * 72)
        print(f"ROUND {round_num}")
        print("=" * 72)

        response = call_api()
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        print(f"Assistant content: {message.content!r}")

        if not tool_calls:
            print(f"\n✅ FINAL (round {round_num}):")
            print(message.content or "(empty)")
            return message.content

        # --- Multi-tool handling: execute ALL tool_calls from this response ---
        count = len(tool_calls)
        print(f"\n📦 Simultaneous tool calls in ONE response: {count}")
        for tc in tool_calls:
            print(f"   • {tc.function.name}({tc.function.arguments})")

        messages.append(assistant_message_from_response(message))
        # ExecuteTool = ExecuteTool()

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as error:
                result = f"Error: invalid JSON arguments: {error}"
            else:
                print(f"\n▶ -----> Executing {tc.function.name}...")
                result = ExecuteTool().execute_tool(tc.function.name, args)
                print(f"  Result: {result[:200]}{'...' if len(result) > 200 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        print()

    return "Max rounds reached without final answer."


def parse_args():
    parser = argparse.ArgumentParser(description="Day 10 — simultaneous tool use")
    parser.add_argument(
        "-q", "--question",
        default=(
            "What time is it in India, what is sum of 10, 15, 20, 15 and 40,"
            "and search the web for how to build a AI agent?"
        ),
        help="User question (designed to trigger multiple tools at once)",
    )
    parser.add_argument(
        "--tool-choice",
        choices=["auto", "none", "required"],
        default="auto",
        help="Groq tool_choice mode (default: auto)",
    )
    parser.add_argument(
        "--force-tool",
        choices=sorted(TOOL_NAMES),
        help="Force a specific tool (overrides --tool-choice)",
    )
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent(
        user_message=args.question,
        tool_choice_mode=args.tool_choice,
        force_tool=args.force_tool,
        max_rounds=args.max_rounds,
    )
