import os
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

class GroqExplorer:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.results = []
    
    def send_api_call(self, call_number, messages, system=None, temperature=1.0, max_tokens=100):
        """Send a single API call with specified parameters"""
        print(f"\n{'='*60}")
        print(f"CALL #{call_number}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")
        print(f"System Prompt: {system if system else 'None'}")
        print(f"Message: {messages[-1]['content'][:100]}...")
        print(f"{'='*60}")
        
        try:
            # Prepare messages (Groq uses OpenAI-compatible chat format)
            api_messages = []
            
            # Add system message if provided
            if system:
                api_messages.append({"role": "system", "content": system})
            
            # Add conversation messages
            api_messages.extend(messages)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            print(f"✅ Response: {response_text}")
            print(f"📊 Tokens Used: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
            
            # Store result
            self.results.append({
                'call_number': call_number,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'system': system,
                'response': response_text,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            })
            
            return response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_experiments(self):
        """Run 10 API calls with variations"""
        
        base_message = "Explain the concept of artificial intelligence in 2-3 sentences."
        
        # ============================================================
        # CALL 1: Baseline - Default settings
        # ============================================================
        self.send_api_call(
            call_number=1,
            messages=[{"role": "user", "content": base_message}],
            temperature=1.0,
            max_tokens=100
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 2: Low temperature (deterministic)
        # ============================================================
        self.send_api_call(
            call_number=2,
            messages=[{"role": "user", "content": base_message}],
            temperature=0.0,
            max_tokens=100
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 3: High temperature (creative)
        # ============================================================
        self.send_api_call(
            call_number=3,
            messages=[{"role": "user", "content": base_message}],
            temperature=1.5,
            max_tokens=100
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 4: Short response (low max_tokens)
        # ============================================================
        self.send_api_call(
            call_number=4,
            messages=[{"role": "user", "content": base_message}],
            temperature=0.7,
            max_tokens=20
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 5: Long response (high max_tokens)
        # ============================================================
        self.send_api_call(
            call_number=5,
            messages=[{"role": "user", "content": base_message}],
            temperature=0.7,
            max_tokens=300
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 6: With system prompt - Expert role
        # ============================================================
        self.send_api_call(
            call_number=6,
            messages=[{"role": "user", "content": base_message}],
            system="You are a world-renowned AI researcher with 20 years of experience. Provide precise, technical explanations.",
            temperature=0.7,
            max_tokens=100
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 7: With system prompt - Beginner-friendly
        # ============================================================
        self.send_api_call(
            call_number=7,
            messages=[{"role": "user", "content": base_message}],
            system="You are a friendly teacher explaining AI to a 10-year-old. Use simple language and fun analogies.",
            temperature=0.8,
            max_tokens=100
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 8: With system prompt - Creative writer
        # ============================================================
        self.send_api_call(
            call_number=8,
            messages=[{"role": "user", "content": base_message}],
            system="You are a creative writer. Explain AI using poetic language and vivid metaphors.",
            temperature=0.9,
            max_tokens=120
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 9: Multi-turn conversation
        # ============================================================
        self.send_api_call(
            call_number=9,
            messages=[
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is computer systems that can perform tasks normally requiring human intelligence."},
                {"role": "user", "content": "Give me a more detailed explanation with examples."}
            ],
            temperature=0.7,
            max_tokens=150
        )
        time.sleep(0.5)
        
        # ============================================================
        # CALL 10: Extreme combination (high temperature + long response + system prompt)
        # ============================================================
        self.send_api_call(
            call_number=10,
            messages=[{"role": "user", "content": base_message}],
            system="You are a philosopher exploring the nature of consciousness and intelligence.",
            temperature=1.8,
            max_tokens=200
        )
    
    def generate_report(self):
        """Generate a summary report of all calls"""
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY REPORT")
        print("="*60)
        
        print(f"\nTotal Calls: {len(self.results)}")
        print(f"Model Used: {self.model}")
        
        print("\n📈 Results by Call:")
        print("-"*60)
        for r in self.results:
            print(f"\nCall #{r['call_number']}:")
            print(f"  Temperature: {r['temperature']}")
            print(f"  Max Tokens: {r['max_tokens']}")
            print(f"  System Prompt: {r['system'][:50] + '...' if r['system'] else 'None'}")
            print(f"  Output Length: {len(r['response'])} characters")
            print(f"  Tokens Used: {r['input_tokens']} + {r['output_tokens']} = {r['input_tokens'] + r['output_tokens']}")
            print(f"  Response Preview: {r['response'][:150]}...")
        
        # Token usage summary
        total_input = sum(r['input_tokens'] for r in self.results)
        total_output = sum(r['output_tokens'] for r in self.results)
        print("\n💳 Total Token Usage:")
        print(f"  Total Input Tokens: {total_input}")
        print(f"  Total Output Tokens: {total_output}")
        print(f"  Total: {total_input + total_output}")
        print("  Estimated Cost: $0.00 (Groq free tier)")
        
        # Temperature effect analysis
        print("\n🌡️ Temperature Effects:")
        low_temp = [r for r in self.results if r['temperature'] <= 0.5]
        high_temp = [r for r in self.results if r['temperature'] >= 1.5]
        
        if low_temp:
            print(f"  Low Temperature (≤0.5): {len(low_temp)} calls")
            print(f"    Example: {low_temp[0]['response'][:100]}...")
        
        if high_temp:
            print(f"  High Temperature (≥1.5): {len(high_temp)} calls")
            print(f"    Example: {high_temp[0]['response'][:100]}...")

def main():
    """Main execution function"""
    print("🚀 Starting Groq API Exploration")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will send 10 API calls with variations in:")
    print("  - Temperature (0.0 to 1.8)")
    print("  - Max tokens (20 to 300)")
    print("  - System prompts (expert, teacher, creative writer, philosopher)")
    print("  - Conversation length (single turn vs multi-turn)")
    print(f"  - Model: llama-3.3-70b-versatile")
    
    # Check for API key
    api_key = os.getenv("GORQ_API_KEY")
    if not api_key:
        print("\n❌ ERROR: GROQ_API_KEY (or GORQ_API_KEY) not found in .env file!")
        print("Please add your Groq API key to the .env file:")
        print("GORQ_API_KEY=gsk-your-key-here")
        return
    
    # Confirm before proceeding
    response = input("\n⚠️  This will use Groq free-tier API quota. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return
    
    explorer = GroqExplorer()
    explorer.run_experiments()
    explorer.generate_report()
    
    print("\n✅ Experiment complete!")
    print("Check the results above for insights on how different parameters affect Groq's responses.")

if __name__ == "__main__":
    main()