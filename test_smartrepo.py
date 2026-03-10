import requests

BASE_URL = "http://127.0.0.1:8000"

def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# ── Test 1: Server Health ──────────────────────────
print_section("TEST 1: Server Health")
try:
    res = requests.get(BASE_URL, timeout=5)
    print(f"✅ Server is reachable (status {res.status_code})")
except Exception as e:
    print(f"❌ Server not reachable: {e}")
    print("   → Make sure uvicorn is running: uvicorn backend.main:app --reload")
    exit(1)

# ── Test 2: Query (uses existing FAISS index) ──────
print_section("TEST 2: Query Endpoint")

questions = [
    "What does this repository do?",
    "How is the FAISS index built?",
    "What file types does the parser support?",
]

for question in questions:
    print(f"\n📌 Question: {question}")
    try:
        res = requests.post(
            f"{BASE_URL}/api/query",
            data={"question": question},
            timeout=60
        )
        if res.status_code == 200:
            data = res.json()
            print(f"✅ Answer: {data['answer']}")
            print(f"   Sources: {data['sources']}")
        elif res.status_code == 400:
            print(f"⚠️  No index found: {res.json()['detail']}")
            print("   → Upload a repo first (see Test 3 below)")
        else:
            print(f"❌ Error {res.status_code}: {res.json()}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

# ── Test 3: Upload a GitHub Repo (optional) ────────
print_section("TEST 3: GitHub Upload (optional)")
TEST_REPO = "https://github.com/tiangolo/fastapi"  # small public repo

run_upload = input("\nRun GitHub upload test? This will re-index. (y/n): ").strip().lower()
if run_upload == "y":
    print(f"⏳ Uploading {TEST_REPO} ...")
    try:
        res = requests.post(
            f"{BASE_URL}/api/upload/github",
            data={"url": TEST_REPO},
            timeout=300
        )
        if res.status_code == 200:
            print(f"✅ {res.json()['message']}")
        else:
            print(f"❌ Upload failed {res.status_code}: {res.json()}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
else:
    print("⏭️  Skipped.")

print_section("ALL TESTS DONE")