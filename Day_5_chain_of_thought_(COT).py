import os
import time

from dotenv import load_dotenv
from groq import APIStatusError, Groq
from typing import List

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_API_RETRIES = 3


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
    
    def run_experiment(self, questions: List[str], max_tokens=1000):
        """Run both prompting styles on each question"""
        for idx, question in enumerate(questions, 1):
            print(f"\n{'='*70}")
            print(f"📝 Question {idx}: {question}")
            print('='*70)
            
            # Style 1: "Think step by step"
            print("\n🔹 Style: Think Step by Step")
            result1 = self.prompt_step_by_step(question, max_tokens)
            
            # Wait for 2 seconds to avoid rate limits
            time.sleep(2)  # avoid rate limits directly by using time.sleep  
            
            # Style 2: XML tags
            print("\n🔹 Style: XML Tags")
            result2 = self.prompt_xml_tags(question, max_tokens)


            
            self.results.append({
                "question": question,
                "step_by_step": result1,
                "xml_tags": result2
            })
    
    def prompt_step_by_step(self, question, max_tokens):
        """Style 1: Direct 'think step by step' instruction"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves problems by thinking step by step."},
            {"role": "user", "content": f"Solve the following problem step by step:\n\n{question}"}
        ]
        
        answer, tokens = self._call_api(messages, max_tokens)

        print(f"Response:\n{answer}\n")
        print(f"Tokens used: {tokens}")

        return {"answer": answer, "tokens": tokens}

    def prompt_xml_tags(self, question, max_tokens):
        """Style 2: Use XML tags <thinking> and <answer>"""
        system = """
        You must solve problems by first showing your reasoning inside <thinking> tags,
        then provide the final answer inside <answer> tags.
        
        Format:
        <thinking>Your step-by-step reasoning here...</thinking>
        <answer>Your final answer here...</answer>
        """
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Solve this problem:\n\n{question}"}
        ]
        
        answer, tokens = self._call_api(messages, max_tokens)

        print(f"Response:\n{answer}\n")
        print(f"Tokens used: {tokens}")

        return {"answer": answer, "tokens": tokens}



class CoTAnalyzer:
    def __init__(self, results):
        self.results = results
    
    def extract_final_answer(self, text):
        """Extract the final answer from XML tags if present"""
        import re
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
    
    def count_reasoning_steps(self, text):
        """Count steps or bullet points"""
        steps = text.count('step') + text.count('\n') + text.count('1.') + text.count('2.')
        return steps
    
    def generate_report(self):
        print("\n" + "="*70)
        print("📊 EXPERIMENT ANALYSIS")
        print("="*70)
        
        total_steps = 0
        total_step_tokens = 0
        total_xml_tokens = 0
        
        for i, res in enumerate(self.results, 1):
            print(f"\n📌 Question {i}: {res['question']}")
            
            step_ans = res["step_by_step"]["answer"]
            xml_ans = res["xml_tags"]["answer"]
            
            # Analyse step_by_step
            step_tokens = res["step_by_step"]["tokens"]
            total_step_tokens += step_tokens
            
            # Analyse XML
            xml_tokens = res["xml_tags"]["tokens"]
            total_xml_tokens += xml_tokens
            
            # Extract XML answer
            xml_final = self.extract_final_answer(xml_ans)
            xml_steps = xml_ans.count('<thinking>')  # just a rough measure
            
            print(f"  Step-by-step: {step_tokens} tokens")
            print(f"  XML tags: {xml_tokens} tokens")
            print(f"  Final answer (XML): {xml_final[:100]}...")
            
            total_steps += xml_steps
        
        # Summary
        print("\n" + "-"*70)
        print("📈 SUMMARY STATISTICS")
        print("-"*70)
        
        if total_step_tokens:
            avg_step_tokens = total_step_tokens / len(self.results)
        else:
            avg_step_tokens = 0
        
        if total_xml_tokens:
            avg_xml_tokens = total_xml_tokens / len(self.results)
        else:
            avg_xml_tokens = 0
        
        
        print(f"Average tokens per question:")
        print(f"  Step-by-step: {avg_step_tokens:.0f} tokens")
        print(f"  XML tags: {avg_xml_tokens:.0f} tokens")
        print(f"  Difference: {avg_step_tokens - avg_xml_tokens:.0f} tokens")
        
        print("\n💡 OBSERVATIONS:")
        if avg_step_tokens < avg_xml_tokens:
            print("  - XML tags use more tokens (extra tag tokens)")
        else:
            print("  - Step-by-step uses more tokens (less structured)")
        
        print("  - XML tags provide cleaner separation of reasoning and answer")
        print("  - Step-by-step is more natural for the model")


if __name__ == "__main__":

    # Define a set of test questions
    questions = [
        "If John has 5 apples, gives 2 to Mary, then buys 3 more, and eats 1, how many apples does he have?",
        "What is the sum of the first 10 prime numbers?",
        "A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. What is the total distance?"
    ]

    # Run experiment
    exp = CoTExperiment()
    exp.run_experiment(questions)

    # Analyze
    analyzer = CoTAnalyzer(exp.results)
    analyzer.generate_report()

    # some more problems:

    # "Solve: 3x + 7 = 22, find x.",
    # "If you flip a coin 3 times, what is the probability of getting at least 2 heads?",
    # "Logic puzzle: All cats are animals. Some animals are pets. Can we conclude that all cats are pets? Explain."