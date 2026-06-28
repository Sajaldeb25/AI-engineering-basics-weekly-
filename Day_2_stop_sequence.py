import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_api_key():
    """Accept GROQ_API_KEY or GORQ_API_KEY — matches .env and other Day 2 scripts."""
    return os.getenv("GORQ_API_KEY")


class StopSequenceStudy:
    def __init__(self):
        self.client = Groq(api_key=get_groq_api_key())
        # mixtral-8x7b-32768 was decommissioned; use current Groq model
        self.model = "llama-3.3-70b-versatile"
        self.results = []

    def generate_with_stops(self, prompt, stop_sequences=None, max_tokens=200, temperature=0.7):
        """Generate text with stop sequences"""
        print("\n" + "=" * 70)
        print(f"🔴 STOP SEQUENCE: {stop_sequences if stop_sequences else 'None'}")
        print("=" * 70)

        print(f"\n📝 Prompt: {prompt}")
        print("\n🤖 Response:")
        print("-" * 70)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )

            response_text = response.choices[0].message.content or ""

            print(f"\n{response_text}")
            print(f"\n📊 Tokens used: {response.usage.total_tokens}")
            print(f"📌 Stop reason: {response.choices[0].finish_reason}")

            result = {
                "prompt": prompt,
                "stop_sequences": stop_sequences,
                "response": response_text,
                "tokens": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }
            self.results.append(result)

            return result

        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None

    def study_stop_sequences(self):
        """Comprehensive study of stop sequences"""
        print("🚀 STUDY: STOP SEQUENCES")
        print("=" * 70)
        print("\nThis study demonstrates how stop sequences control AI responses")
        print("and why they're critical for building agents.\n")

        self.generate_with_stops(
            prompt="Explain artificial intelligence in detail.",
            stop_sequences=None,
            max_tokens=200,
        )
        time.sleep(0.5)

        self.generate_with_stops(
            prompt="List 3 benefits of AI:\n1.",
            stop_sequences=["\n"],
            max_tokens=200,
        )
        time.sleep(0.5)

        self.generate_with_stops(
            prompt="Complete this sentence: 'The future of AI is '",
            stop_sequences=["."],
            max_tokens=100,
        )
        time.sleep(0.5)

        self.generate_with_stops(
            prompt="Write a recipe for spaghetti.",
            stop_sequences=["\n\n", "!", "?"],
            max_tokens=200,
        )
        time.sleep(0.5)

        self.generate_with_stops(
            prompt="Generate a JSON object with user info: name, age, city. Respond in JSON format.",
            stop_sequences=["}\n"],
            max_tokens=200,
        )
        time.sleep(0.5)

        self.generate_with_stops(
            prompt="List 5 programming languages:\n1. Python\n2.",
            stop_sequences=["\n\n"],
            max_tokens=100,
        )
        time.sleep(0.5)

        self.generate_with_stops(
            prompt="What is the capital of France? The capital is ",
            stop_sequences=["."],
            max_tokens=50,
        )
        time.sleep(0.5)

    def analyze_results(self):
        """Analyze the stop sequence study results"""
        print("\n" + "=" * 70)
        print("📊 ANALYSIS: STOP SEQUENCE STUDY")
        print("=" * 70)

        if not self.results:
            print("\n⚠️  No successful API calls to analyze.")
            print("Check your GORQ_API_KEY in .env and that the model is available.")
            return

        print("\n🎯 Key Observations:\n")

        for result in self.results:
            text = result.get("response") or ""
            preview = text[:100] + ("..." if len(text) > 100 else "")
            tail = text[-20:] if text else "(empty)"

            print(f"Stop: {result['stop_sequences'] if result['stop_sequences'] else 'None'}")
            print(f"  Finish reason: {result['finish_reason']}")
            print(f"  Tokens: {result['tokens']}")
            print(f"  Response preview: {preview}")
            print(f"  Stops at: '{tail}'\n")

        stop_hits = sum(1 for r in self.results if r["finish_reason"] == "stop")
        length_hits = sum(1 for r in self.results if r["finish_reason"] == "length")

        print("-" * 70)
        print("💡 INSIGHTS:\n")
        print(f"  Successful calls: {len(self.results)}")
        print(f"  Stopped by stop sequence (finish_reason='stop'): {stop_hits}")
        print(f"  Hit max_tokens (finish_reason='length'): {length_hits}")
        print()
        print("1. Stop sequences give you PRECISE CONTROL over where the AI stops")
        print("2. Different stop sequences create different response structures")
        print("3. Multiple stop sequences provide flexibility")
        print("4. Without stops, responses can continue until max_tokens")
        print("5. For agents, stop sequences enable structured, predictable outputs")


def main():
    if not get_groq_api_key():
        print("\n❌ ERROR: GROQ_API_KEY (or GORQ_API_KEY) not found in .env file!")
        print("Please add your Groq API key to the .env file:")
        print("GORQ_API_KEY=gsk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return

    study = StopSequenceStudy()
    study.study_stop_sequences()
    study.analyze_results()


if __name__ == "__main__":
    main()
