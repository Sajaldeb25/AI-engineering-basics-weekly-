import os
import json
from dotenv import load_dotenv
from groq import Groq
import tiktoken

load_dotenv()

class GroqTokenCounter:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GORQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"  # Groq's free model
        
        # Pricing per 1M tokens (Groq rates — free tier has no charge)
        self.pricing = {
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        }
        
        # Initialize tokenizer for accurate counting
        try:
            # Use cl100k_base for general counting
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to simple word-based counting
            self.encoder = None
    
    def count_tokens(self, text):
        """Count tokens in a text string"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
    
    def count_messages_tokens(self, messages):
        """Count tokens in a list of messages"""
        total_tokens = 0
        
        for message in messages:
            # Count tokens in role and content
            role_tokens = self.count_tokens(message.get("role", ""))
            content_tokens = self.count_tokens(message.get("content", ""))
            total_tokens += role_tokens + content_tokens
            
            # Add extra tokens for formatting (approximate)
            total_tokens += 4  # Per message overhead
        
        # Add system prompt tokens if present
        total_tokens += 2  # System prompt overhead
        
        return total_tokens
    
    def estimate_cost(self, messages, max_tokens=100):
        """Estimate the cost of an API call"""
        # Count input tokens
        input_tokens = self.count_messages_tokens(messages)
        
        # Estimated output tokens (max_tokens)
        output_tokens = max_tokens
        
        # Get pricing for the model
        model_pricing = self.pricing.get(self.model, {"input": 0.5, "output": 0.5})
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_1k_input": model_pricing["input"] / 1000,
            "cost_per_1k_output": model_pricing["output"] / 1000
        }
    
    def send_and_track(self, messages, token_len, temperature=0.7, max_tokens=100):
        """Send API call and track token usage"""
        print("\n" + "="*60)
        print("📤 SENDING API CALL")
        print("="*60)
        
        # Show message preview
        print("\n📝 Messages:")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            print(f"  {role}: {content}...")
        
        # Estimate cost before sending
        print("\n💰 Estimated Cost:")
        estimate = self.estimate_cost(messages, max_tokens)
        print(f"  Input Tokens: {estimate['input_tokens']}")
        print(f"  Output Tokens: {estimate['output_tokens']} (max)")
        print(f"  Total Tokens: {estimate['total_tokens']}")
        print(f"  Estimated Cost: ${estimate['total_cost']:.6f}")
        print(f"  Cost per 1K input: ${estimate['cost_per_1k_input']:.4f}")
        print(f"  Cost per 1K output: ${estimate['cost_per_1k_output']:.4f}")
        
        # Send actual API call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Get actual usage
            actual_input = response.usage.prompt_tokens
            actual_output = response.usage.completion_tokens
            actual_total = response.usage.total_tokens
            
            # Calculate actual cost
            model_pricing = self.pricing[self.model]
            actual_cost = (actual_input / 1_000_000) * model_pricing["input"] + \
                         (actual_output / 1_000_000) * model_pricing["output"]
            
            print("\n✅ Response Received:")
            print(f"  Content: {response.choices[0].message.content[:token_len]}...")
            
            print("\n📊 Actual Usage:")
            print(f"  Input Tokens: {actual_input}")
            print(f"  Output Tokens: {actual_output}")
            print(f"  Total Tokens: {actual_total}")
            print(f"  Actual Cost: ${actual_cost:.6f}")
            
            # Compare estimate vs actual
            print("\n📈 Estimate vs Actual:")
            print(f"  Input: {estimate['input_tokens']} estimated vs {actual_input} actual")
            print(f"  Output: {estimate['output_tokens']} estimated vs {actual_output} actual")
            print(f"  Cost: ${estimate['total_cost']:.6f} estimated vs ${actual_cost:.6f} actual")
            print(f"  Accuracy: {((1 - abs(estimate['total_cost'] - actual_cost) / (actual_cost + 0.000001)) * 100):.1f}%")
            
            return {
                "response": response,
                "estimated": estimate,
                "actual": {
                    "input_tokens": actual_input,
                    "output_tokens": actual_output,
                    "total_tokens": actual_total,
                    "cost": actual_cost
                }
            }
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None

def main():
    """Run token counting experiments"""
    print("🚀 Groq Token Counter & Cost Estimator")
    print("="*60)
    
    # Check API key
    if not os.getenv("GORQ_API_KEY"):
        print("\n❌ ERROR: GORQ_API_KEY not found in .env file!")
        print("Please add your Groq API key to the .env file:")
        print("GORQ_API_KEY=gsk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return
    
    counter = GroqTokenCounter()
    
    # Test Case 1: Simple message
    messages_simple = [
        {"role": "user", "content": "Explain artificial intelligence in 2-3 sentences."}
    ]
    
    print("\n📝 TEST 1: Simple Message")
    result1 = counter.send_and_track(messages_simple, token_len=100, max_tokens=100)
    
    # Test Case 2: With system prompt
    messages_system = [
        {"role": "system", "content": "You are a technical AI expert. Provide precise, detailed answers."},
        {"role": "user", "content": "Explain artificial intelligence in 2-3 sentences."}
    ]
    
    print("\n📝 TEST 2: With System Prompt")
    result2 = counter.send_and_track(messages_system, token_len=150, max_tokens=150)
    
    # Test Case 3: Multi-turn conversation
    messages_multi = [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is computer systems that mimic human intelligence."},
        {"role": "user", "content": "Give me examples of AI applications."}
    ]
    
    print("\n📝 TEST 3: Multi-turn Conversation")
    result3 = counter.send_and_track(messages_multi, token_len=500, max_tokens=500)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY REPORT")
    print("="*60)
    
    results = [r for r in [result1, result2, result3] if r]
    
    if results:
        total_estimated = sum(r['estimated']['total_cost'] for r in results)
        total_actual = sum(r['actual']['cost'] for r in results)
        total_tokens = sum(r['actual']['total_tokens'] for r in results)
        
        print(f"\nTotal API Calls: {len(results)}")
        print(f"Total Tokens Used: {total_tokens}")
        print(f"Total Estimated Cost: ${total_estimated:.6f}")
        print(f"Total Actual Cost: ${total_actual:.6f}")
        print(f"Average Cost per Call: ${total_actual/len(results):.6f}")
        
        print("\n💡 Cost Breakdown:")
        print(f"  - Groq's free tier: 30 requests per minute")
        print(f"  - Model: {counter.model}")
        print(f"  - Input:  ${counter.pricing[counter.model]['input']}/1M tokens")
        print(f"  - Output: ${counter.pricing[counter.model]['output']}/1M tokens")
        print(f"  - Your total cost: ${total_actual:.6f} (free tier eligible)")
        
        print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()