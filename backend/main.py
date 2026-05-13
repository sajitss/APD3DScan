import os
import uuid
import shutil
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import processing modules
from processing.marker_detection import process_scan_markers
from processing.segmentation import process_segmentation
from processing.reconstruction import run_reconstruction

app = FastAPI(title="3D Scan Prototype Backend")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Global status storage (In-memory for prototype)
scan_statuses = {}

class ScanStatus(BaseModel):
    scan_id: str
    status: str
    progress: int
    quality_report: dict = None

def run_full_pipeline(scan_id: str, images_dir: str):
    try:
        scan_statuses[scan_id]["status"] = "marker_detection_started"
        
        # 1. Marker Detection & Quality Reporting
        marker_output_dir = os.path.join(OUTPUTS_DIR, scan_id, "markers")
        report = process_scan_markers(scan_id, images_dir, marker_output_dir)
        scan_statuses[scan_id]["quality_report"] = report
        scan_statuses[scan_id]["status"] = "marker_detection_completed"
        
        # 2. Segmentation / Masking
        scan_statuses[scan_id]["status"] = "reconstruction_started"
        masked_output_dir = os.path.join(OUTPUTS_DIR, scan_id, "masked_images")
        process_segmentation(images_dir, report["per_image_results"], masked_output_dir)
        
        # 3. Reconstruction (Stub)
        recon_output_dir = os.path.join(OUTPUTS_DIR, scan_id, "reconstruction")
        run_reconstruction(scan_id, masked_output_dir, recon_output_dir)
        
        scan_statuses[scan_id]["status"] = "reconstruction_completed"
    except Exception as e:
        scan_statuses[scan_id]["status"] = f"failed: {str(e)}"

@app.post("/api/upload-scan")
async def upload_scan(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    scan_id = str(uuid.uuid4())
    scan_dir = os.path.join(UPLOADS_DIR, scan_id, "images")
    os.makedirs(scan_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(scan_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    scan_statuses[scan_id] = {
        "scan_id": scan_id,
        "status": "uploaded",
        "progress": 0,
        "quality_report": None
    }
    
    # Start processing in background
    background_tasks.add_task(run_full_pipeline, scan_id, scan_dir)
    
    return {"scan_id": scan_id, "message": "Upload successful, processing started."}

@app.get("/api/scan-status/{scan_id}")
async def get_scan_status(scan_id: str):
    if scan_id not in scan_statuses:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan_statuses[scan_id]

@app.get("/api/result/{scan_id}")
async def get_result(scan_id: str):
    if scan_id not in scan_statuses:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    result_path = os.path.join(OUTPUTS_DIR, scan_id)
    if not os.path.exists(result_path):
         raise HTTPException(status_code=404, detail="Result files not found")
         
    return {
        "scan_id": scan_id,
        "quality_report": scan_statuses[scan_id].get("quality_report"),
        "status": scan_statuses[scan_id].get("status")
    }

# Serve Frontend
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
