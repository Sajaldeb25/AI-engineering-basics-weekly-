import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from groq import Groq

# Load environment variables
load_dotenv()

def test_openai():
    """Test OpenAI connection"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello, OpenAI!'"}
            ],
            max_tokens=10
        )
        print("✓ OpenAI connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ OpenAI error: {e}")
        return False

def test_anthropic():
    """Test Anthropic connection"""
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[
                {"role": "user", "content": "Say 'Hello, Anthropic!'"}
            ]
        )
        print("✓ Anthropic connection successful!")
        print(f"Response: {response.content[0].text}")
        return True
    except Exception as e:
        print(f"✗ Anthropic error: {e}")
        return False

def test_gemini():
    """Test Gemini connection"""
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Hello, Gemini!'",
        )
        print("✓ Gemini connection successful!")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"✗ Gemini error: {e}")
        return False

def test_groq():
    """Test Groq connection"""
    try:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")
        client = Groq(api_key=api_key)
        content = "Tell me about sheikh hasina of Bangladesh in 2 sentence."
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
        )
        print("✓ Groq connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Groq error: {e}")
        return False


if __name__ == "__main__":
    print("Testing API connections...\n")

    groq_ok = test_groq()

    print("\nResults:")
    print(f"Groq: {'✓' if groq_ok else '✗'}")
