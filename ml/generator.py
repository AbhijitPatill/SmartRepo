import requests
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

SYSTEM_PROMPT = """You are SmartRepo, an expert code analysis AI. You have been given exact source code from a software repository.

Your rules:
- Answer confidently and in detail — explain the full flow, not just a one-liner
- NEVER use uncertain language like "maybe", "perhaps", "I think", "I'm not sure", "it appears", "it seems"
- If the answer is in the code, state it as fact with clear explanation
- Walk through the logic step by step when explaining flows or architecture
- Always reference specific file names and function names from the context
- If something is not in the provided context, say: "This information is not in the indexed files."
- Be technical but friendly — you are a helpful senior developer explaining to a teammate"""

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = ""
    for chunk in chunks:
        context += f"\n--- File: {chunk['path']} ---\n{chunk['content']}\n"
    return f"Code context:\n{context}\n\nQuestion: {query}"

def generate_answer(query: str, chunks: list[dict]) -> str:
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, chunks)}
        ],
        "max_tokens": 1024,
        "temperature": 0.2  # low but not robotic
    }
    try:
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload, timeout=60)
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        elif "error" in result:
            return f"Model error: {result['error']}"
        return "No response from model."
    except Exception as e:
        return f"Error: {str(e)}"