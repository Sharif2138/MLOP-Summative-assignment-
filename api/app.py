from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
from prediction import predict, load_model
from upload_new_data import upload_new_data
from retrain import retrain_pipeline
import os
import uuid

app = FastAPI()


@app.get("/")
def root():
    return {"message": "API running"}


@app.on_event("startup")
def startup_event():
    load_model()


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)

    file_location = f"temp/{uuid.uuid4()}_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        class_name, confidence = predict(file_location)
        return {"class_name": class_name, "confidence": confidence}
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


@app.post("/upload")
async def upload_data(files: List[UploadFile] = File(...)):
    upload_new_data(files)
    return {"message": "Upload successful", "files": len(files)}


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_pipeline)
    return {"message": "Retraining started"}
