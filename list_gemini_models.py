"""
List available Gemini models for your API key
------------------------------------------------
Model names change fairly often, and availability can differ by account
and region. Instead of guessing, this hits Google's own "list models"
endpoint so you can see exactly which model names YOUR key can use right
now, and copy the correct one into config.py.

Usage:
    python list_gemini_models.py
"""

import requests
import config

url = f"{config.LLM_API_BASE_URL.rstrip('/')}/models"
headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()
data = response.json()

print("Models available to your API key:\n")
for model in data.get("data", []):
    print(f"  {model['id']}")
