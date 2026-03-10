from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import tempfile
import os
import sys
import json
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from ml.parser import parse_repo
from ml.embedder import get_embeddings
from ml.retriever import build_index, search, index_exists, get_indexed_files
from ml.generator import generate_answer

router = APIRouter()


def progress_event(stage: str, percent: int, message: str):
    data = json.dumps({"stage": stage, "percent": percent, "message": message})
    return f"data: {data}\n\n"


@router.post("/upload/zip")
async def upload_zip(file: UploadFile = File(...)):
    async def stream():
        try:
            yield progress_event("saving", 5, "Saving uploaded file...")
            await asyncio.sleep(0.1)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp.write(await file.read())
            tmp.close()

            yield progress_event("parsing", 15, "Extracting and parsing files...")
            await asyncio.sleep(0.1)

            chunks = parse_repo(tmp.name, is_zip=True)
            yield progress_event("parsed", 35, f"Parsed {len(chunks)} chunks from repository...")
            await asyncio.sleep(0.1)

            yield progress_event("embedding", 45, "Generating embeddings (this takes a moment)...")
            await asyncio.sleep(0.1)

            chunks, embeddings = get_embeddings(chunks)
            yield progress_event("embedded", 80, f"Embeddings ready for {len(chunks)} chunks...")
            await asyncio.sleep(0.1)

            yield progress_event("indexing", 88, "Storing in ChromaDB...")
            await asyncio.sleep(0.1)

            build_index(chunks, embeddings)
            os.unlink(tmp.name)

            yield progress_event("done", 100, f"Indexed {len(chunks)} chunks successfully.")

        except Exception as e:
            yield progress_event("error", 0, str(e))

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })


@router.post("/upload/github")
async def upload_github(url: str = Form(...)):
    async def stream():
        try:
            yield progress_event("cloning", 5, "Cloning repository from GitHub...")
            await asyncio.sleep(0.1)

            chunks = parse_repo(url, is_zip=False)
            yield progress_event("parsed", 35, f"Parsed {len(chunks)} chunks from repository...")
            await asyncio.sleep(0.1)

            yield progress_event("embedding", 45, "Generating embeddings (this takes a moment)...")
            await asyncio.sleep(0.1)

            chunks, embeddings = get_embeddings(chunks)
            yield progress_event("embedded", 80, f"Embeddings ready for {len(chunks)} chunks...")
            await asyncio.sleep(0.1)

            yield progress_event("indexing", 88, "Storing in ChromaDB...")
            await asyncio.sleep(0.1)

            build_index(chunks, embeddings)

            yield progress_event("done", 100, f"Indexed {len(chunks)} chunks successfully.")

        except Exception as e:
            yield progress_event("error", 0, str(e))

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })


@router.get("/context")
async def get_context():
    """Returns indexed file list and AI-generated sample questions."""
    try:
        if not index_exists():
            return {"files": [], "questions": []}

        files = get_indexed_files()

        sample_q_prompt = f"""Based on these files from a software repository: {', '.join(files[:20])}
Generate exactly 4 short, specific questions a developer might ask about this codebase.
Return ONLY a JSON array of 4 strings, nothing else. Example: ["Q1?", "Q2?", "Q3?", "Q4?"]"""

        dummy_chunks = [{"path": f, "content": ""} for f in files[:5]]
        raw = generate_answer(sample_q_prompt, dummy_chunks)

        import re
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        questions = json.loads(match.group()) if match else [
            "What does this repository do?",
            "How is the main logic structured?",
            "What dependencies does this project use?",
            "How do I run this project?"
        ]

        return {"files": files[:10], "questions": questions[:4]}

    except Exception as e:
        return {"files": [], "questions": [], "error": str(e)}


@router.post("/query")
async def query(question: str = Form(...)):
    try:
        if not index_exists():
            raise HTTPException(status_code=400, detail="No repository indexed yet. Please upload first.")

        from ml.embedder import model
        query_embedding = model.encode(question)
        relevant_chunks = search(query_embedding, query_text=question)

        answer = generate_answer(question, relevant_chunks)

        return {
            "answer": answer,
            "sources": list(dict.fromkeys([c["path"] for c in relevant_chunks]))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))