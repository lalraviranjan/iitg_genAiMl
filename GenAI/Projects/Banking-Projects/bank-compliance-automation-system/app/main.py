from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import json
from workflow import run_workflow
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploaded_files"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Workflow API is running."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ["pdf", "docx"]:
            raise HTTPException(status_code=400, detail="Only PDF or DOCX files are allowed.")

        unique_id = uuid.uuid4().hex[:8]
        file_name = f"{unique_id}.{file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, file_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call workflow
        json_file_name = run_workflow(file_path, unique_id)

        return {"message": "File processed successfully", "file_id": unique_id, "result_file": json_file_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result/{file_id}")
def get_result(file_id: str):
    try:
        json_file_path = os.path.join(RESULT_FOLDER, f"compliance_{file_id}.json")
        print("*" * 50)
        print(f"JSON file path: {json_file_path}")
        # json_file_path = json_file_path.replace("\\", "/")

        if not os.path.exists(json_file_path):
            raise HTTPException(status_code=404, detail="Result file not found.")

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
