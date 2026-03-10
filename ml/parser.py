import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from git import Repo

SUPPORTED_EXTENSIONS = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".md", ".txt", ".json", ".html", ".css", ".sol"]
IGNORE_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"}
IGNORE_FILES = {"package-lock.json", "yarn.lock", "poetry.lock", "package.json"}

def extract_zip(zip_path: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_dir)
    return tmp_dir

def clone_repo(github_url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    Repo.clone_from(github_url, tmp_dir)
    return tmp_dir

def get_files(repo_dir: str) -> list[dict]:
    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for filename in filenames:
            if filename in IGNORE_FILES:
                continue
            ext = Path(filename).suffix
            if ext in SUPPORTED_EXTENSIONS:
                filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(filepath, repo_dir)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        files.append({
                            "path": relative_path,
                            "content": content,
                            "extension": ext
                        })
                except Exception:
                    continue
    return files

def chunk_file(file: dict, max_chars: int = 1500) -> list[dict]:
    content = file["content"]
    chunks = []
    lines = content.split("\n")
    current_chunk = []
    current_len = 0

    for line in lines:
        current_chunk.append(line)
        current_len += len(line)
        if current_len >= max_chars:
            chunks.append({
                "path": file["path"],
                "content": "\n".join(current_chunk),
                "extension": file["extension"]
            })
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunks.append({
            "path": file["path"],
            "content": "\n".join(current_chunk),
            "extension": file["extension"]
        })

    return chunks

def parse_repo(source: str, is_zip: bool = True) -> list[dict]:
    if is_zip:
        repo_dir = extract_zip(source)
    else:
        repo_dir = clone_repo(source)

    files = get_files(repo_dir)
    all_chunks = []
    for file in files:
        all_chunks.extend(chunk_file(file))

    shutil.rmtree(repo_dir, ignore_errors=True)
    return all_chunks