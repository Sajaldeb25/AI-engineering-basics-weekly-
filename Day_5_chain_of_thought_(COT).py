import argparse
import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from groq import APIStatusError, Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_API_RETRIES = 3
RATE_LIMIT_DELAY_SECONDS = 2

DEFAULT_QUESTIONS = [
    "If John has 5 apples, gives 2 to Mary, then buys 3 more, and eats 1, how many apples does he have?",
    "What is the sum of the first 10 prime numbers?",
    "A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. What is the total distance?",
    "Solve: 3x + 7 = 22, find x.",
    "If you flip a coin 3 times, what is the probability of getting at least 2 heads?",
    "Logic puzzle: All cats are animals. Some animals are pets. Can we conclude that all cats are pets? Explain.",
]


def get_groq_api_key():
    return os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")


def create_groq_client():
    api_key = get_groq_api_key()
    if not api_key:
        raise SystemExit(
            "❌ ERROR: GROQ_API_KEY (or GORQ_API_KEY) not found in .env"
        )
    return Groq(api_key=api_key)


class CoTExperiment:
    def __init__(self, client=None):
        self.client = client or create_groq_client()
        self.model = DEFAULT_MODEL
        self.results = []

    def _call_api(self, messages, max_tokens):
        """Call Groq with exponential backoff on rate-limit/API errors."""
        wait_seconds = 1
        last_error = None

        for attempt in range(MAX_API_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens if response.usage else 0
                return content or "", tokens
            except APIStatusError as error:
                last_error = error
                if attempt < MAX_API_RETRIES - 1:
                    print(f"⚠️  API error, retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    wait_seconds *= 2
                else:
                    raise
            except Exception:
                raise

        if last_error:
            raise last_error
        return "", 0

    def _safe_prompt(self, prompt_fn, question, max_tokens):
        try:
            return prompt_fn(question, max_tokens)
        except Exception as error:
            print(f"❌ Prompt failed: {error}")
            return {"answer": "", "tokens": 0, "error": str(error)}

    def run_experiment(self, questions: List[str], max_tokens=1000):
        """Run both prompting styles on each question."""
        for idx, question in enumerate(questions, 1):
            print(f"\n{'='*70}")
            print(f"📝 Question {idx}: {question}")
            print("=" * 70)

            print("\n🔹 Style: Think Step by Step")
            result1 = self._safe_prompt(self.prompt_step_by_step, question, max_tokens)

            time.sleep(RATE_LIMIT_DELAY_SECONDS)

            print("\n🔹 Style: XML Tags")
            result2 = self._safe_prompt(self.prompt_xml_tags, question, max_tokens)

            self.results.append({
                "question": question,
                "step_by_step": result1,
                "xml_tags": result2,
            })

            if idx < len(questions):
                time.sleep(RATE_LIMIT_DELAY_SECONDS)

    def prompt_step_by_step(self, question, max_tokens):
        """Style 1: Direct 'think step by step' instruction."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that solves problems by thinking step by step.",
            },
            {
                "role": "user",
                "content": f"Solve the following problem step by step:\n\n{question}",
            },
        ]

        answer, tokens = self._call_api(messages, max_tokens)

        print(f"Response:\n{answer}\n")
        print(f"Tokens used: {tokens}")

        return {"answer": answer, "tokens": tokens}

    def prompt_xml_tags(self, question, max_tokens):
        """Style 2: Use XML tags <thinking> and <answer>."""
        system = textwrap.dedent("""
            You must solve problems by first showing your reasoning inside <thinking> tags,
            then provide the final answer inside <answer> tags.

            Format:
            <thinking>Your step-by-step reasoning here...</thinking>
            <answer>Your final answer here...</answer>
        """).strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Solve this problem:\n\n{question}"},
        ]

        answer, tokens = self._call_api(messages, max_tokens)

        print(f"Response:\n{answer}\n")
        print(f"Tokens used: {tokens}")

        return {"answer": answer, "tokens": tokens}

    def save_results(self, output_path: str):
        Path(output_path).write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n💾 Results saved to {output_path}")


class CoTAnalyzer:
    def __init__(self, results):
        self.results = results

    def extract_tag_content(self, text: str, tag: str) -> Optional[str]:
        """Extract content from XML-style tags (case-insensitive, tolerates missing close tag)."""
        if not text:
            return None

        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        open_pattern = rf"<{tag}>(.*)"
        match = re.search(open_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def extract_final_answer(self, text):
        """Extract the final answer from XML tags if present."""
        extracted = self.extract_tag_content(text, "answer")
        return extracted if extracted is not None else (text or "").strip()

    def extract_thinking(self, text):
        """Extract reasoning block from XML tags if present."""
        return self.extract_tag_content(text, "thinking")

    def count_reasoning_steps(self, text):
        """Estimate reasoning steps from numbered lines and 'Step N' labels."""
        if not text:
            return 0

        numbered_lines = len(re.findall(r"(?m)^\s*\d+\.", text))
        step_labels = len(re.findall(r"(?i)\bstep\s+\d+", text))
        return numbered_lines + step_labels

    def has_xml_structure(self, text):
        return bool(
            re.search(r"<\s*thinking\s*>", text or "", re.IGNORECASE)
            and re.search(r"<\s*answer\s*>", text or "", re.IGNORECASE)
        )

    def generate_report(self):
        print("\n" + "=" * 70)
        print("📊 EXPERIMENT ANALYSIS")
        print("=" * 70)

        if not self.results:
            print("\nNo results to analyze.")
            return

        total_step_tokens = 0
        total_xml_tokens = 0
        total_step_reasoning_steps = 0
        total_xml_reasoning_steps = 0
        xml_structured_count = 0

        for i, res in enumerate(self.results, 1):
            print(f"\n📌 Question {i}: {res['question']}")

            step_ans = res["step_by_step"]["answer"]
            xml_ans = res["xml_tags"]["answer"]

            step_tokens = res["step_by_step"]["tokens"]
            xml_tokens = res["xml_tags"]["tokens"]
            total_step_tokens += step_tokens
            total_xml_tokens += xml_tokens

            step_steps = self.count_reasoning_steps(step_ans)
            thinking_text = self.extract_thinking(xml_ans) or xml_ans
            xml_steps = self.count_reasoning_steps(thinking_text)
            total_step_reasoning_steps += step_steps
            total_xml_reasoning_steps += xml_steps

            xml_final = self.extract_final_answer(xml_ans)
            if self.has_xml_structure(xml_ans):
                xml_structured_count += 1

            preview = xml_final[:100] + ("..." if len(xml_final) > 100 else "")
            print(f"  Step-by-step: {step_tokens} tokens, ~{step_steps} reasoning steps")
            print(f"  XML tags: {xml_tokens} tokens, ~{xml_steps} reasoning steps")
            print(f"  Final answer (XML): {preview}")

            if res["step_by_step"].get("error"):
                print(f"  ⚠️  Step-by-step error: {res['step_by_step']['error']}")
            if res["xml_tags"].get("error"):
                print(f"  ⚠️  XML tags error: {res['xml_tags']['error']}")

        print("\n" + "-" * 70)
        print("📈 SUMMARY STATISTICS")
        print("-" * 70)

        n = len(self.results)
        avg_step_tokens = total_step_tokens / n
        avg_xml_tokens = total_xml_tokens / n
        avg_step_reasoning = total_step_reasoning_steps / n
        avg_xml_reasoning = total_xml_reasoning_steps / n
        token_diff = avg_step_tokens - avg_xml_tokens

        print("Average tokens per question:")
        print(f"  Step-by-step: {avg_step_tokens:.0f} tokens")
        print(f"  XML tags: {avg_xml_tokens:.0f} tokens")
        print(f"  Difference: {token_diff:.0f} tokens")

        print("\nAverage reasoning steps per question:")
        print(f"  Step-by-step: {avg_step_reasoning:.1f}")
        print(f"  XML tags: {avg_xml_reasoning:.1f}")

        print("\n💡 OBSERVATIONS:")
        if token_diff > 10:
            print("  - Step-by-step used more tokens on average (less structured output)")
        elif token_diff < -10:
            print("  - XML tags used more tokens on average (tag overhead + explicit structure)")
        else:
            print("  - Both styles used a similar number of tokens on average")

        if avg_xml_reasoning > avg_step_reasoning:
            print("  - XML-tagged responses showed more explicit reasoning steps")
        elif avg_step_reasoning > avg_xml_reasoning:
            print("  - Step-by-step responses showed more explicit reasoning steps")
        else:
            print("  - Both styles produced a similar amount of explicit reasoning")

        structured_ratio = xml_structured_count / n
        if structured_ratio >= 0.8:
            print(f"  - XML tags produced valid <thinking>/<answer> structure in {xml_structured_count}/{n} runs")
        elif structured_ratio > 0:
            print(f"  - XML tags only produced full structure in {xml_structured_count}/{n} runs")
        else:
            print("  - XML tags did not produce expected tag structure in any run")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare chain-of-thought prompting styles on Groq."
    )
    parser.add_argument(
        "-q", "--question",
        action="append",
        dest="questions",
        help="Question to run (repeatable). Uses defaults if omitted.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Max completion tokens per API call (default: 1000).",
    )
    parser.add_argument(
        "-o", "--output",
        default="cot_results.json",
        help="Path to save JSON results (default: cot_results.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    questions = args.questions or DEFAULT_QUESTIONS

    exp = CoTExperiment()
    exp.run_experiment(questions, max_tokens=args.max_tokens)
    exp.save_results(args.output)

    analyzer = CoTAnalyzer(exp.results)
    analyzer.generate_report()
