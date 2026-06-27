# Day 01 — AI API Exploration

A Python learning project for calling multiple AI providers (OpenAI, Anthropic, Google Gemini, and Groq), testing API connections, and experimenting with model parameters like temperature, max tokens, and system prompts.

## Project Structure

```
day01/
├── main.py                      # Multi-provider AI client (OpenAI + Anthropic)
├── test_setup.py                # Test API connections (currently runs Groq)
├── ten_run_with_diff_prams.py   # 10 Groq API calls with different parameters
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create this yourself — not committed to git)
├── .gitignore                   # Ignores .env, venv, __pycache__
└── venv01/                      # Virtual environment (create locally)
```

## Prerequisites

- **Python 3.10+** (tested with Python 3.12)
- **pip** (Python package manager)
- At least one valid **API key** (see [API Keys](#api-keys) below)

Check your Python version:

```bash
python3 --version
```

## Environment Setup

### 1. Clone or navigate to the project

```bash
cd /path/to/day01
```

### 2. Create a virtual environment

```bash
python3 -m venv venv01
```

### 3. Activate the virtual environment

**Linux / macOS:**

```bash
source venv01/bin/activate
```

**Windows (Command Prompt):**

```cmd
venv01\Scripts\activate
```

**Windows (PowerShell):**

```powershell
venv01\Scripts\Activate.ps1
```

When active, your terminal prompt will show `(venv01)`.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package        | Purpose                          |
|----------------|----------------------------------|
| `openai`       | OpenAI API client                |
| `anthropic`    | Anthropic (Claude) API client    |
| `google-genai` | Google Gemini API client         |
| `groq`         | Groq API client (free tier)      |
| `python-dotenv`| Load variables from `.env` file  |

### 5. Create the `.env` file

Create a file named `.env` in the project root. **Never commit this file to git.**

```env
# Paid providers (require billing/credits)
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Free-tier providers
GEMINI_API_KEY=your-gemini-key
GORQ_API_KEY=gsk-your-groq-key
```

> **Note:** This project uses `GORQ_API_KEY` (typo). The code also accepts the standard name `GROQ_API_KEY`. Use either one.

#### Where to get API keys

| Provider   | Sign up / API key URL                              | Cost        |
|------------|----------------------------------------------------|-------------|
| **Groq**   | https://console.groq.com                           | Free tier   |
| **Gemini** | https://aistudio.google.com/apikey                 | Free tier   |
| **OpenAI** | https://platform.openai.com/api-keys               | Paid        |
| **Anthropic** | https://console.anthropic.com/settings/keys     | Paid        |

**Recommended for learning without payment:** use **Groq** (`GORQ_API_KEY`).

## Running the Scripts

Make sure the virtual environment is activated before running any script:

```bash
source venv01/bin/activate   # Linux/macOS
```

---

### `test_setup.py` — Test API connections

Quick health check for your API keys. Currently configured to test **Groq**.

```bash
python3 test_setup.py
```

**Expected output (success):**

```
Testing API connections...

✓ Groq connection successful!
Response: ...

Results:
Groq: ✓
```

**What it tests:**

| Function          | Provider  | Requires key        | Status in project        |
|-------------------|-----------|---------------------|--------------------------|
| `test_groq()`     | Groq      | `GORQ_API_KEY`      | Active (runs by default) |
| `test_gemini()`   | Gemini    | `GEMINI_API_KEY`    | Available, not called    |
| `test_openai()`   | OpenAI    | `OPENAI_API_KEY`    | Available, not called    |
| `test_anthropic()`| Anthropic | `ANTHROPIC_API_KEY` | Available, not called    |

To test another provider, edit the `if __name__ == "__main__"` block at the bottom of `test_setup.py` and call the function you want (e.g. `test_gemini()`).

---

### `main.py` — Multi-provider AI client

Demonstrates a reusable `AIClient` class that calls **OpenAI** and **Anthropic**.

```bash
python3 main.py
```

**What it does:**

- Sends `"What is Python?"` to OpenAI (`gpt-4`)
- Sends the same question to Anthropic (`claude-3-opus-20240229`)
- Prints both responses

**Requirements:** Valid `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` with active billing. These are paid services — if you have no credits, this script will fail with quota/billing errors.

**Use as a module:**

```python
from main import AIClient

client = AIClient()
print(client.ask_openai("Your question here"))
print(client.ask_anthropic("Your question here"))
```

---

### `ten_run_with_diff_prams.py` — Groq parameter experiments

Runs **10 API calls** to Groq with different settings to show how parameters affect responses.

```bash
python3 ten_run_with_diff_prams.py
```

Type `y` when prompted to confirm.

**What it experiments with:**

| Call | Variation                                      |
|------|------------------------------------------------|
| 1    | Baseline (temperature 1.0)                     |
| 2    | Low temperature 0.0 (deterministic)          |
| 3    | High temperature 1.5 (creative)              |
| 4    | Short response (max_tokens=20)                 |
| 5    | Long response (max_tokens=300)                 |
| 6    | System prompt: AI expert                       |
| 7    | System prompt: teacher for kids                |
| 8    | System prompt: creative writer                 |
| 9    | Multi-turn conversation                        |
| 10   | High temperature + system prompt + long output |

**Model used:** `llama-3.3-70b-versatile`

**Requirements:** `GORQ_API_KEY` (or `GROQ_API_KEY`) in `.env`

At the end, the script prints a summary report with token usage and response previews.

---

## Quick Start (Groq only — free)

If you only have a Groq key and want to get running fast:

```bash
cd day01
python3 -m venv venv01
source venv01/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
GORQ_API_KEY=gsk-your-key-here
```

Test the connection:

```bash
python3 test_setup.py
```

Run the 10-call experiment:

```bash
python3 ten_run_with_diff_prams.py
```

## Troubleshooting

### `python: command not found`

Use `python3` instead:

```bash
python3 test_setup.py
```

### `ModuleNotFoundError: No module named 'groq'`

Activate the virtual environment and install dependencies:

```bash
source venv01/bin/activate
pip install -r requirements.txt
```

### OpenAI: `429 insufficient_quota`

Your OpenAI account has no remaining credits. Add billing at https://platform.openai.com/account/billing or use Groq instead.

### Anthropic: `credit balance is too low`

Your Anthropic account needs credits. Add billing at https://console.anthropic.com/settings/billing or use Groq instead.

### Gemini: `429 RESOURCE_EXHAUSTED`

Free-tier quota is used up. Wait for the daily reset, create a new key, or switch to Groq.

### Groq: connection works in `test_setup.py` but fails in experiments

Check that `GORQ_API_KEY` is set in `.env` and that you confirmed with `y` when running `ten_run_with_diff_prams.py`.

## Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`
- **Never paste API keys** into `requirements.txt`, source code, or README files
- **Do not add personal notes** (email, bank info, etc.) inside `.env`
- If a key is accidentally exposed, revoke it immediately in the provider's dashboard and create a new one

## Dependencies Reference

Full list from `requirements.txt`:

```
openai>=1.0.0
anthropic>=0.7.0
python-dotenv>=1.0.0
google-genai>=1.0.0
groq>=0.4.0
```

## Suggested Learning Path

1. Set up the virtual environment and `.env` file
2. Run `test_setup.py` to verify your Groq key works
3. Run `ten_run_with_diff_prams.py` and compare how temperature and system prompts change responses
4. Read `main.py` to understand how a reusable multi-provider client is structured
5. Optionally add Groq support to `main.py` for a fully free workflow
