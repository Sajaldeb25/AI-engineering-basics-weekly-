"""
Day 4 — Multi-turn conversational loop with three memory strategies:
  - sliding: keep last N user/assistant turns
  - token: truncate oldest messages when context budget is exceeded
  - summary: compress old history into a summary when threshold is hit
"""

import os
import time

import tiktoken
from dotenv import load_dotenv
from groq import APIStatusError, Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_CONTEXT_TOKENS = 8000
DEFAULT_SUMMARY_THRESHOLD = 6000
DEFAULT_MAX_COMPLETION_TOKENS = 200
MESSAGE_OVERHEAD_TOKENS = 4


def get_groq_api_key():
    return os.getenv("GORQ_API_KEY")


def get_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_message_tokens(messages, encoder):
    """Estimate tokens for a chat message list (includes per-message overhead)."""
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if encoder:
            total += len(encoder.encode(role + content))
        else:
            total += len(role + content) // 4
        total += MESSAGE_OVERHEAD_TOKENS
    return total


class BaseChat:
    """Shared Groq client, API retry logic, and safe turn storage."""

    def __init__(
        self,
        client,
        model=DEFAULT_MODEL,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    ):
        self.client = client
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.system_content = "You are a helpful assistant."
        self.history = []
        self.encoder = get_encoder()

    def clear_history(self):
        self.history = []

    def count_tokens(self, messages):
        return count_message_tokens(messages, self.encoder)

    def _system_message(self, extra_context=""):
        content = self.system_content
        if extra_context:
            content = f"{content}\n\n{extra_context}"
        return {"role": "system", "content": content}

    def _call_api(self, messages):
        """Call Groq with exponential backoff on API/rate-limit errors."""
        wait_seconds = 1
        last_error = None

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=self.max_completion_tokens,
                )
                return response.choices[0].message.content or ""
            except APIStatusError as error:
                last_error = error
                if attempt < 2:
                    print(f"⚠️  API error, retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    wait_seconds *= 2
                else:
                    raise
            except Exception:
                raise

        if last_error:
            raise last_error
        return ""

    def _save_turn(self, user_input, assistant_msg):
        """Store a complete user/assistant pair only after a successful API call."""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_msg})

    def chat(self, user_input):
        raise NotImplementedError


class SlidingWindowChat(BaseChat):
    """Keep only the last N user/assistant pairs in memory."""

    def __init__(self, client, max_turns=10, **kwargs):
        super().__init__(client, **kwargs)
        self.max_turns = max_turns

    def _trim_history(self):
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def chat(self, user_input):
        messages = self._system_message(), *self.history, {
            "role": "user",
            "content": user_input,
        }
        messages = list(messages)

        assistant_msg = self._call_api(messages)
        self._save_turn(user_input, assistant_msg)
        self._trim_history()
        return assistant_msg


class TokenManagedChat(BaseChat):
    """Drop oldest messages when estimated context tokens exceed the budget."""

    def __init__(self, client, max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS, **kwargs):
        super().__init__(client, **kwargs)
        self.max_context_tokens = max_context_tokens

    def _truncate_messages(self, messages):
        system_msg = messages[0]
        rolling = messages[1:]

        while rolling and self.count_tokens([system_msg, *rolling]) > self.max_context_tokens:
            rolling.pop(0)

        return [system_msg, *rolling]

    def chat(self, user_input):
        messages = [
            self._system_message(),
            *self.history,
            {"role": "user", "content": user_input},
        ]
        messages = self._truncate_messages(messages)

        assistant_msg = self._call_api(messages)
        self._save_turn(user_input, assistant_msg)
        return assistant_msg


class SummarizingChat(BaseChat):
    """Summarize old conversation into a separate memory field (not history)."""

    def __init__(
        self,
        client,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        summary_threshold=DEFAULT_SUMMARY_THRESHOLD,
        **kwargs,
    ):
        super().__init__(client, **kwargs)
        self.max_context_tokens = max_context_tokens
        self.summary_threshold = summary_threshold
        self.summary = ""

    def clear_history(self):
        super().clear_history()
        self.summary = ""

    def _format_history(self):
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)

    def _summarize_history(self):
        if not self.history:
            return

        print("📝 Context full — summarizing older conversation...")
        summary_prompt = (
            "Summarize the following conversation in a few sentences. "
            "Keep key facts and context needed for future replies.\n\n"
            f"Conversation:\n{self._format_history()}"
        )

        summary_text = self._call_api([{"role": "user", "content": summary_prompt}])
        self.summary = summary_text.strip()
        self.history = []

    def _build_messages(self, user_input):
        extra = ""
        if self.summary:
            extra = f"Previous conversation summary:\n{self.summary}"

        return [
            self._system_message(extra_context=extra),
            *self.history,
            {"role": "user", "content": user_input},
        ]

    def chat(self, user_input):
        messages = self._build_messages(user_input)

        if self.count_tokens(messages) > self.summary_threshold:
            self._summarize_history()
            messages = self._build_messages(user_input)

        if self.count_tokens(messages) > self.max_context_tokens:
            self.history = self.history[-4:]

        assistant_msg = self._call_api(messages)
        self._save_turn(user_input, assistant_msg)
        return assistant_msg


class ChatApp:
    STRATEGIES = ("token", "sliding", "summary")

    def __init__(self, strategy="token", client=None):
        self.strategy = strategy
        self.client = client or Groq(api_key=get_groq_api_key())

        if strategy == "sliding":
            self.chat = SlidingWindowChat(self.client, max_turns=10)
        elif strategy == "summary":
            self.chat = SummarizingChat(self.client)
        else:
            self.chat = TokenManagedChat(self.client)

    def run(self):
        print("🤖 Multi-turn chat started")
        print(f"   Strategy: {self.strategy}")
        print(f"   Model: {DEFAULT_MODEL}")
        print("   Commands: exit | history | clear")
        print("-" * 50)

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            command = user_input.lower()
            if command == "exit":
                print("👋 Goodbye!")
                break
            if command == "history":
                self.show_history()
                continue
            if command == "clear":
                self.chat.clear_history()
                print("🧹 History cleared.")
                continue

            try:
                response = self.chat.chat(user_input)
                print(f"AI: {response}")
            except APIStatusError as error:
                print(f"❌ Groq API error: {error}")
            except Exception as error:
                print(f"❌ Error: {error}")

    def show_history(self):
        print("\n--- Chat History ---")
        if hasattr(self.chat, "summary") and self.chat.summary:
            preview = self.chat.summary[:100]
            suffix = "..." if len(self.chat.summary) > 100 else ""
            print(f"summary: {preview}{suffix}")

        for msg in self.chat.history:
            content = msg.get("content") or ""
            preview = content[:100]
            suffix = "..." if len(content) > 100 else ""
            print(f"{msg['role']}: {preview}{suffix}")
        print("-------------------")


if __name__ == "__main__":
    if not get_groq_api_key():
        print("❌ ERROR: GORQ_API_KEY (or GROQ_API_KEY) not found in .env")
        raise SystemExit(1)

    import sys

    strategy = sys.argv[1] if len(sys.argv) > 1 else "token"
    if strategy not in ChatApp.STRATEGIES:
        print(f"Unknown strategy '{strategy}'. Choose: {', '.join(ChatApp.STRATEGIES)}")
        raise SystemExit(1)

    ChatApp(strategy=strategy).run()
