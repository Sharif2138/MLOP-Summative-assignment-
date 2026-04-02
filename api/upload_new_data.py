# upload_new_data.py
from supabase import create_client
from fastapi import UploadFile
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def upload_new_data(files: List[UploadFile]) -> None:
    supabase = create_client(
        supabase_url=os.getenv("supabase_project_url"),
        supabase_key=os.getenv("supabase_anon_key"),
    )

    BUCKET = "training_data"
    uploaded = 0
    skipped = 0

    for file in files:
        if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            print(f"[skipped]  {file.filename} — not an allowed image type")
            skipped += 1
            continue

        cloud_path = f"training_data/new_data/{file.filename}"
        contents = await file.read()

        try:
            supabase.storage.from_(BUCKET).upload(
                cloud_path,
                contents,
                {"content-type": "application/octet-stream", "upsert": "true"},
            )
            print(f"[uploaded] {cloud_path}")
            uploaded += 1
        except Exception as e:
            print(f"[error]    {cloud_path} — {e}")
            skipped += 1

    print(f"\nDone. {uploaded} uploaded, {skipped} failed.")
