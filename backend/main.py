from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

print("Starting SmartRepo...", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)

try:
    print("Importing routes...", flush=True)
    from api.routes import router
    print("Routes imported OK", flush=True)
except Exception as e:
    print(f"ROUTES IMPORT FAILED: {e}", flush=True)
    sys.exit(1)

app = FastAPI(title="SmartRepo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"status": "SmartRepo backend is running"}