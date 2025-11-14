from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from .inference import analyze_blade_image



app = FastAPI()



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)



@app.get("/")

def root():

    return {"status": "BladeGuard API running"}



@app.post("/analyze-blade")

async def analyze_blade(

    file: UploadFile = File(...),

    blade_type: str = Form("UNKNOWN")  # "TPI" or "LM" preferred

):

    image_bytes = await file.read()

    result = analyze_blade_image(image_bytes, blade_type=blade_type)

    return result


@app.get("/view-heatmap/{heatmap_filename}")
async def view_heatmap(heatmap_filename: str):
    """Serve heatmap image file"""
    heatmap_path = os.path.join("results", heatmap_filename)
    if os.path.exists(heatmap_path):
        return FileResponse(heatmap_path, media_type="image/png")
    return {"error": "Heatmap not found"}


@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    """HTML viewer for heatmap visualization"""
    viewer_path = os.path.join(os.path.dirname(__file__), "static", "viewer.html")
    if os.path.exists(viewer_path):
        with open(viewer_path, "r") as f:
            return f.read()
    return "<html><body><h1>Viewer not found</h1></body></html>"
