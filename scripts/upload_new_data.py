from supabase import create_client
import os
import shutil

def upload_new_data(folder_path):
    supabase = create_client(
        supabase_url=os.getenv("supabase_project_url"),
        supabase_key=os.getenv("supabase_anon_key")
    )

    