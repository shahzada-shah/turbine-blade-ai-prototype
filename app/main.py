from fastapi import FastAPI, UploadFile, File, Form

from fastapi.middleware.cors import CORSMiddleware

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
