import os
import time
from dotenv import load_dotenv
from groq import Groq
import tiktoken

load_dotenv()


def get_groq_api_key():
    return os.getenv("GORQ_API_KEY") # get the API key from the .env file


class GroqStreaming:
    def __init__(self):
        self.client = Groq(api_key=get_groq_api_key())
        self.model = "llama-3.3-70b-versatile"

        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

        self.streaming_results = []

    def count_tokens(self, text):
        """Count tokens in text (requires tiktoken; otherwise rough estimate)"""
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text) // 4
    
    def stream_response(self, messages, temperature=0.7, max_tokens=200, show_metrics=True):
        """
        Stream response from Groq with real-time chunk display
        
        Args:
            messages: List of message objects
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            show_metrics: Display timing and chunk metrics
        """
        print("\n" + "="*70)
        print("📡 STREAMING RESPONSE")
        print("="*70)
        
        # Show prompt
        print("\n📝 Prompt:")
        for msg in messages:
            if msg["role"] == "user":
                print(f"  👤 User: {msg['content']}")
            elif msg["role"] == "system":
                print(f"  ⚙️ System: {msg['content']}")
        
        print("\n" + "-"*70)
        print("🤖 Response (streaming):")
        print("-"*70)
        print("\n", end="")
        
        # Metrics tracking
        start_time = time.time()  # starting time 
        chunks_received = 0  # number of chunks received
        char_count = 0  # number of characters received
        full_response = ""  # full response
        chunk_arrival_times = []  # time of each chunk arrival
        chunk_intervals = []  # interval between each chunk arrival

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue

                print(content, end="", flush=True)

                full_response += content # add the content to the full response
                char_count += len(content) # add the number of characters to the character count
                chunks_received += 1 # add 1 to the number of chunks received

                now = time.time() - start_time
                if chunk_arrival_times:
                    chunk_intervals.append(now - chunk_arrival_times[-1])
                chunk_arrival_times.append(now)

                if chunks_received % 5 == 0:
                    print(f"\033[90m [{chunks_received} chucontentnks]\033[0m", end="", flush=True)

            print("\n")

            elapsed_time = time.time() - start_time

            chunks_per_second = chunks_received / elapsed_time if elapsed_time > 0 else 0
            chars_per_second = char_count / elapsed_time if elapsed_time > 0 else 0
            estimated_tokens = self.count_tokens(full_response)

            result = {
                "full_response": full_response,
                "chunks_received": chunks_received,
                "estimated_tokens": estimated_tokens,
                "char_count": char_count,
                "elapsed_time": elapsed_time,
                "chunks_per_second": chunks_per_second,
                "chars_per_second": chars_per_second,
                "chunk_arrival_times": chunk_arrival_times,
                "chunk_intervals": chunk_intervals,
            }
            self.streaming_results.append(result)
            
            if show_metrics:
                self.display_metrics(result)
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None
    
    def display_metrics(self, result):
        """Display streaming metrics"""
        print("\n" + "="*70)
        print("📊 STREAMING METRICS")
        print("="*70)

        arrival_times = result["chunk_arrival_times"]
        intervals = result["chunk_intervals"]

        print(f"\n⏱️  Time to first chunk: {arrival_times[0] if arrival_times else 0:.3f}s")
        print(f"⏱️  Total time: {result['elapsed_time']:.3f}s")
        print(f"📝 Total characters: {result['char_count']}")
        print(f"📦 Chunks received: {result['chunks_received']}")
        print(f"🔢 Estimated tokens (tiktoken): {result['estimated_tokens']}")
        print(f"⚡ Speed: {result['chunks_per_second']:.2f} chunks/sec")
        print(f"⚡ Speed: {result['chars_per_second']:.2f} chars/sec")

        if intervals:
            print("\n⏱️  Chunk Interval Analysis (time between chunks):")

            print("  First 10 intervals:")
            for i, delta in enumerate(intervals[:10], 1):
                print(f"    Chunk {i:2d}: {delta:.4f}s")

            if len(intervals) > 10:
                print("  Last 5 intervals:")
                for i, delta in enumerate(intervals[-5:], len(intervals) - 4):
                    print(f"    Chunk {i:2d}: {delta:.4f}s")

            avg_interval = sum(intervals) / len(intervals)
            print(f"\n  Average time between chunks: {avg_interval:.4f}s")

            if len(intervals) > 1:
                first_half = intervals[: len(intervals) // 2]
                second_half = intervals[len(intervals) // 2 :]

                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)

                print(f"  Average interval (first half): {first_avg:.4f}s")
                print(f"  Average interval (second half): {second_avg:.4f}s")

                if second_avg > first_avg * 1.1:
                    print("  ⚠️  Slower in second half")
                elif second_avg < first_avg * 0.9:
                    print("  ✅ Faster in second half")
    
    def stream_with_colors(self, messages, temperature=0.7, max_tokens=200):
        """Stream with colored output for better visualization"""
        print("\n" + "="*70)
        print("🎨 STREAMING WITH COLORS")
        print("="*70)
        
        # Display prompt in color
        print("\n📝 \033[94mUser:\033[0m", messages[-1]["content"])
        
        print("\n" + "-"*70)
        print("🤖 \033[92mResponse (colored streaming):\033[0m")
        print("-"*70)
        print("\n", end="")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
            colors = ['\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']
            color_index = 0
            chunks_received = 0

            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue

                color = colors[color_index % len(colors)]
                print(f"{color}{content}\033[0m", end="", flush=True)

                chunks_received += 1
                color_index += 1

                if chunks_received % 10 == 0:
                    print(f"\033[90m [{chunks_received} chunks]\033[0m", end="", flush=True)

            print("\n")

        except Exception as e:
            print(f"\n❌ Error: {e}")

def demonstrate_streaming_use_cases():
    """Demonstrate different streaming use cases"""
    
    print("🚀 GROQ STREAMING DEMONSTRATION")
    print("="*70)
    print("\nThis demonstrates real-time chunk streaming with different use cases")
    
    if not get_groq_api_key():
        print("\n❌ ERROR: GROQ_API_KEY (or GORQ_API_KEY) not found in .env file!")
        print("Please add your Groq API key to the .env file:")
        print("GORQ_API_KEY=gsk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return
    
    streamer = GroqStreaming()
    
    # Use Case 1: Simple streaming
    print("\n📌 USE CASE 1: Basic Streaming")
    print("-"*70)
    messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms in 3 sentences."}
    ]
    streamer.stream_response(messages, max_tokens=100)
    
    time.sleep(1)
    
    # Use Case 2: Streaming with system prompt
    print("\n📌 USE CASE 2: Streaming with System Prompt")
    print("-"*70)
    messages = [
        {"role": "system", "content": "You are a helpful AI that explains complex topics simply."},
        {"role": "user", "content": "What is artificial intelligence?"}
    ]
    streamer.stream_response(messages, max_tokens=150)
    
    time.sleep(1)
    
    # Use Case 3: Long-form generation with streaming
    print("\n📌 USE CASE 3: Long-Form Content Streaming")
    print("-"*70)
    messages = [
        {"role": "user", "content": "Write a short story about a time-traveling scientist (100 words)."}
    ]
    streamer.stream_response(messages, max_tokens=200)
    
    time.sleep(1)
    
    # Use Case 4: Colored streaming
    print("\n📌 USE CASE 4: Colored Chunk Streaming")
    print("-"*70)
    messages = [
        {"role": "user", "content": "Write a haiku about programming."}
    ]
    streamer.stream_with_colors(messages, max_tokens=50)
    
    # Summary of all streams
    print("\n" + "="*70)
    print("📊 STREAMING SUMMARY")
    print("="*70)
    
    if streamer.streaming_results:
        total_chunks = sum(r["chunks_received"] for r in streamer.streaming_results)
        total_tokens = sum(r["estimated_tokens"] for r in streamer.streaming_results)
        total_time = sum(r["elapsed_time"] for r in streamer.streaming_results)
        avg_speed = total_chunks / total_time if total_time > 0 else 0

        print(f"\nTotal streams: {len(streamer.streaming_results)}")
        print(f"Total chunks streamed: {total_chunks}")
        print(f"Total estimated tokens: {total_tokens}")
        print(f"Total streaming time: {total_time:.2f}s")
        print(f"Average speed: {avg_speed:.2f} chunks/sec")
        
        print("\n✅ All streaming demonstrations completed!")

if __name__ == "__main__":
    demonstrate_streaming_use_cases()