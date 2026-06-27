import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

class AIClient:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def ask_openai(self, prompt):
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def ask_anthropic(self, prompt):
        response = self.anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

# Usage
if __name__ == "__main__":
    client = AIClient()
    
    response = client.ask_openai("What is Python?")
    print(f"OpenAI: {response}\n")
    
    response = client.ask_anthropic("What is Python?")
    print(f"Anthropic: {response}")