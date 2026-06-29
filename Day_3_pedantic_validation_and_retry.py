import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, APIStatusError
from pydantic import BaseModel, ValidationError

load_dotenv()


def get_groq_api_key():
    return os.getenv("GORQ_API_KEY")


def extract_json(text):
    """Strip markdown fences and parse JSON from model output."""
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)

    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    return json.loads(cleaned)


class Person(BaseModel):
    name: str
    age: int
    city: str
    occupation: str
    hobbies: list[str]


class AIResponseProcessor:
    def __init__(self):
        self.client = Groq(api_key=get_groq_api_key())
        self.model = "llama-3.3-70b-versatile"
        self.max_retries = 3

    def _build_messages(self, user_input, error_feedback=None):
        system_content = (
            "You MUST respond with valid JSON only. "
            "Include these fields: name (string), age (integer), city (string), "
            "occupation (string), hobbies (list of strings). "
            'Example: {"name": "Alice", "age": 30, "city": "NYC", '
            '"occupation": "Engineer", "hobbies": ["coding", "reading"]}. '
            "No extra text, no markdown, ONLY JSON."
        )
        messages = [{"role": "system", "content": system_content}]

        if error_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{user_input}\n\n"
                        f"Your previous response failed validation:\n{error_feedback}\n"
                        "Return corrected JSON only."
                    ),
                }
            )
        else:
            messages.append({"role": "user", "content": user_input})

        return messages

    def _format_validation_errors(self, error):
        lines = []
        for item in error.errors():
            field = item["loc"][0] if item["loc"] else "unknown"
            lines.append(f"- {field}: {item['msg']}")
        return "\n".join(lines)

    def get_validated_response(self, user_input):
        """Get and validate AI response with retries and error feedback."""
        error_feedback = None

        for attempt in range(1, self.max_retries + 1):
            print(f"\n🔄 Attempt {attempt}/{self.max_retries}")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages(user_input, error_feedback),
                    temperature=0.3,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )

                raw_text = response.choices[0].message.content or ""
                print(f"Raw: {raw_text}")

                data = extract_json(raw_text)
                validated_data = Person(**data)

                print("✅ Validated successfully!")
                print(f"Name: {validated_data.name}")
                print(f"Age: {validated_data.age} (type: {type(validated_data.age).__name__})")
                return validated_data

            except json.JSONDecodeError as e:
                print(f"❌ JSON Error: {e}")
                error_feedback = f"Invalid JSON: {e}"

            except ValidationError as e:
                print(f"❌ Pydantic Validation Error: {e}")
                for item in e.errors():
                    field = item["loc"][0] if item["loc"] else "unknown"
                    print(f"  - {field}: {item['msg']}")
                error_feedback = self._format_validation_errors(e)

            except APIStatusError as e:
                print(f"❌ Groq API Error: {e}")
                error_feedback = None

            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                error_feedback = None

            if attempt < self.max_retries:
                if error_feedback:
                    print("Retrying with error feedback...")
                
                # need to implement exponential backoff
                # exponential backoff is a technique to slow down the request rate after a failure
                
                time.sleep(2 ** attempt)

        print("❌ Max retries reached!")
        return None


if __name__ == "__main__":
    if not get_groq_api_key():
        print("❌ ERROR: GORQ_API_KEY (or GROQ_API_KEY) not found in .env")
        raise SystemExit(1)

    processor = AIResponseProcessor()
    result = processor.get_validated_response(
        "Give me a fictional person with name, age, city, occupation, and hobbies"
    )

    if result:
        print("\n✅ Final validated data:")
        print(f"Name: {result.name}")
        print(f"Age: {result.age}")
        print(f"City: {result.city}")
        print(f"Occupation: {result.occupation}")
        print(f"Hobbies: {result.hobbies}")
