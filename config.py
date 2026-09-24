# ---- LLM Configuration ----
# This file is SAFE to commit to GitHub — it contains no real secrets,
# only placeholders and structure.
#
# HOW YOUR REAL API KEY GETS IN, WITHOUT EVER BEING COMMITTED:
#   - Locally: create a file called local_secrets.py (already in
#     .gitignore, never uploaded) in the same folder, with your real
#     key, e.g.:  LLM_API_KEY = "AIzaSy...your real key..."
#     It gets loaded automatically at the bottom of this file.
#   - On Streamlit Cloud: your real key lives in the app's Settings ->
#     Secrets, and app.py overrides these placeholder values with it at
#     startup automatically. You don't need local_secrets.py there.

LLM_PROVIDER = "openai_compatible"

# --- Settings used when LLM_PROVIDER = "openai_compatible" ---
# Examples (uncomment/edit the one you're using in local_secrets.py,
# not here):
#
#   Groq:        LLM_API_BASE_URL = "https://api.groq.com/openai/v1"
#                LLM_MODEL_NAME    = "llama-3.1-8b-instant"
#   OpenRouter:  LLM_API_BASE_URL = "https://openrouter.ai/api/v1"
#                LLM_MODEL_NAME    = "meta-llama/llama-3.1-8b-instruct:free"
#   Gemini:      LLM_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
#                LLM_MODEL_NAME    = "gemini-3.6-flash"
#   Local Ollama: LLM_API_BASE_URL = "http://localhost:11434/v1"
#                 LLM_MODEL_NAME    = "llama3.1"
#                 LLM_API_KEY       = "ollama"

LLM_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
LLM_API_KEY = "placeholder-set-your-real-key-in-local_secrets.py"
LLM_MODEL_NAME = "gemini-3.6-flash"

# Optional — see ask_law_bot.py for what this does. "low" is a good
# starting point for Gemini's thinking models; set to None (or delete
# this line) to go back to the model's own default reasoning behavior.
LLM_REASONING_EFFORT = "low"

# --- Settings used when LLM_PROVIDER = "anthropic" ---
ANTHROPIC_API_KEY = "placeholder-set-your-real-key-in-local_secrets.py"
ANTHROPIC_MODEL_NAME = "claude-haiku-4-5-20251001"

# Load real local overrides if the file exists (it won't on Streamlit
# Cloud, since it's gitignored — that's expected and fine there).
try:
    from local_secrets import *  # noqa: F401,F403
except ImportError:
    pass