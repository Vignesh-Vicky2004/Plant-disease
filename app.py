# app.py
import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from inference_sdk import InferenceHTTPClient
from PIL import Image
from io import BytesIO
import json # Import the json library to pretty-print the result

app = FastAPI()

# This is the dictionary that maps user-friendly names to Roboflow model IDs
MODEL_MAP = {
    "Paddy": "rice-plant-leaf-disease-classification/1",
    "Cassava": "cassava-model/1",
    "Mango": "mango-leaf-disease-2/4",
    "Sugarcane": "sugarcane-leaf-disease/2",
    "Tea": "tea-leaf-plant-diseases/1"
}

# IMPORTANT: This line gets the API key from your hosting environment (Render).
# Do NOT paste your key directly into the code.
API_KEY = os.getenv("API_KEY")

# This is a new endpoint to display the upload form on the main page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    This endpoint returns an HTML form for uploading an image and selecting a plant.
    """
    options = "".join([f"<option value='{key}'>{key}</option>" for key in MODEL_MAP.keys()])
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Detection</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; background-color: #f4f4f9; }}
            h1 {{ color: #333; }}
            form {{ background-color: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            select, input[type=file] {{ width: 100%; padding: 10px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 4px; }}
            input[type=submit] {{ background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }}
            input[type=submit]:hover {{ background-color: #0056b3; }}
            #result {{ margin-top: 20px; padding: 1em; background-color: #e9ecef; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }}
        </style>
    </head>
    <body>
        <h1>🌿 Plant Disease Detection API</h1>
        <p>Select a plant type, upload an image, and click Predict.</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label for="plant">Select Plant Type:</label><br>
            <select id="plant" name="plant">
                {options}
            </select>
            <br><br>
            <label for="image">Upload Image:</label><br>
            <input type="file" id="image" name="image" accept="image/*" required>
            <br><br>
            <input type="submit" value="Predict">
        </form>
        <div id="result-container" style="display:none;">
            <h2>Prediction Result:</h2>
            <pre id="result"></pre>
        </div>
        <script>
            const form = document.querySelector('form');
            form.addEventListener('submit', async (e) => {{
                e.preventDefault();
                const formData = new FormData(form);
                const resultContainer = document.getElementById('result-container');
                const resultPre = document.getElementById('result');
                resultContainer.style.display = 'block';
                resultPre.textContent = 'Analyzing image...';

                const response = await fetch('/predict-json', {{
                    method: 'POST',
                    body: formData,
                }});

                const result = await response.json();
                resultPre.textContent = JSON.stringify(result, null, 2);
            }});
        </script>
    </body>
    </html>
    """

# This is your original endpoint, but renamed to avoid conflicts.
# The HTML form will now submit data to this endpoint.
@app.post("/predict-json")
async def predict_json(plant: str = Form(...), image: UploadFile = File(...)):
    if plant not in MODEL_MAP:
        return {"error": "Invalid plant category"}

    if not API_KEY:
        return {"error": "API_KEY environment variable not set on the server."}

    try:
        image_data = await image.read()
        img = Image.open(BytesIO(image_data))

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=API_KEY
        )

        result = client.infer(img, model_id=MODEL_MAP[plant])
        return result
    except Exception as e:
        return {"error": f"An error occurred during inference: {str(e)}"}
