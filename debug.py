import requests
from dotenv import load_dotenv
import os

load_dotenv("backend/.env")
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

res = requests.post(
    "https://router.huggingface.co/v1/chat/completions",
    headers=HEADERS,
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 50,
        "temperature": 0.3
    },
    timeout=30
)

print(f"Status: {res.status_code}")
print(f"Response: {res.text[:300]}")