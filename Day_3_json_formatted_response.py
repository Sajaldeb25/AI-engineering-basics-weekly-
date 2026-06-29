import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_api_key():
    return os.getenv("GORQ_API_KEY")


def extract_json(text):
    """Strip markdown fences and parse JSON from model output."""
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)

    cleaned = text.strip() # remove whitespace from the text
    # text = '   {"name": "Ethan"}   '
    # cleaned = '{"name": "Ethan"}'


    # Remove ```json ... ``` or ``` ... ``` wrappers
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    return json.loads(cleaned)


class GroqJsonFormatter:
    def __init__(self):
        self.client = Groq(api_key=get_groq_api_key())
        self.model = "llama-3.3-70b-versatile"
        self.max_tokens = 500

    def get_structured_response(self, user_input):
        """Get JSON response from AI"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You MUST respond with valid JSON only. "
                        "Do NOT include markdown, code fences, or explanation text."
                    ),
                },
                {"role": "user", "content": user_input},
            ],
            temperature=0.5,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        json_string = response.choices[0].message.content or ""

        try:
            parsed_data = extract_json(json_string)
            print("\n=======================  Valid JSON received:  ========================\n")
            print(json.dumps(parsed_data, indent=2))
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            print(f"Raw response: {json_string}")
            return None


if __name__ == "__main__":
    if not get_groq_api_key():
        print("❌ ERROR: GORQ_API_KEY (or GROQ_API_KEY) not found in .env")
        raise SystemExit(1)

    formatter = GroqJsonFormatter()
    prompt = (
        "Give me a person's info: name, age, city, hobby, profession, "
        "father name, mother name, married status. Make it fictional."
    )

    formatter.get_structured_response(prompt)
    print("\n\n=======================  Second Request:  ========================\n\n")
    formatter.get_structured_response(prompt)
