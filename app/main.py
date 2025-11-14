from fastapi import FastAPI, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware

from .inference import analyze_blade_image



app = FastAPI()



# CORS (allow all for now)

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

async def analyze_blade(file: UploadFile = File(...)):

    image_bytes = await file.read()

    result = analyze_blade_image(image_bytes)

    return result
