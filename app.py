# app.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from inference_sdk import InferenceHTTPClient
from PIL import Image
from io import BytesIO

app = FastAPI()

MODEL_MAP = {
    "Paddy": "rice-plant-leaf-disease-classification/1",
    "Cassava": "cassava-model/1",
    "Mango": "mango-leaf-disease-2/4",
    "Sugarcane": "sugarcane-leaf-disease/2",
    "Tea": "tea-leaf-plant-diseases/1"
}

API_KEY = os.getenv("API_KEY")

@app.post("/predict")
async def predict(plant: str = Form(...), image: UploadFile = File(...)):
    if plant not in MODEL_MAP:
        return {"error": "Invalid plant category"}
    
    image_data = await image.read()
    img = Image.open(BytesIO(image_data))
    
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )
    
    result = client.infer(img, model_id=MODEL_MAP[plant])
    return result
