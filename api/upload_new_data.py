from supabase import create_client
import os


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def upload_new_data(folder_path: str) -> None:
    supabase = create_client(
        supabase_url=os.getenv("supabase_project_url"),
        supabase_key=os.getenv("supabase_anon_key"),
    )

    BUCKET = "training_data"
    uploaded = 0
    skipped = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            # Skip non-image files 
            if not any(file.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(
                full_path, folder_path).replace("\\", "/")


            cloud_path = f"new_data/{rel_path}"

            with open(full_path, "rb") as f:
                try:
                    supabase.storage.from_(BUCKET).upload(
                        cloud_path,
                        f,
                        # upsert=true prevents errors on duplicate filenames
                        {"content-type": "application/octet-stream", "upsert": "true"},
                    )
                    print(f"[uploaded] {cloud_path}")
                    uploaded += 1
                except Exception as e:
                    print(f"[error]    {cloud_path} — {e}")
                    skipped += 1

    print(f"\nDone. {uploaded} uploaded, {skipped} failed.")
