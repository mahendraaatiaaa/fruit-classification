from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FastAPI server...")

# Initialize FastAPI app
app = FastAPI()

# Define model architecture (SimpleNet2D)
class SimpleNet2D(nn.Module):
    def __init__(self, num_classes, img_height, img_width):
        super(SimpleNet2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define model path and Google Drive URL
MODEL_PATH = os.path.join(project_root, "final-deep-learning", "models", "weights", "model2.pth")
MODEL_URL = "https://drive.google.com/uc?id=1Ff7_W-v9-sgS-88wIzuP8fn-vJJbTJbF"

# Function to download model from Google Drive
import gdown

def download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully.")
    else:
        logger.info("Model already exists locally.")


# Define model instance
num_classes = 10  # Replace with your number of classes
model = SimpleNet2D(num_classes=num_classes, img_height=177, img_width=177)

# Load the model's state_dict
if not os.path.exists(MODEL_PATH):
    logger.info("Model file not found. Downloading model...")
    download_model()

state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

# Define class labels and image dimensions
img_height = 177
img_width = 177
class_labels = ['grape', 'apple', 'starfruit', 'orange', 'kiwi', 'mango', 'pineapple', 'banana', 'watermelon', 'strawberry']

# Correct path to index.html
# Jika folder tidak menggunakan spasi
index_file_path = os.path.join(project_root, "final-deep-learning", "docs", "index.html")

# Debugging output
print(f"Calculated index file path: {index_file_path}")
print(f"Index file exists: {os.path.exists(index_file_path)}")


# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
def read_index():
    try:
        # Mengakses file index.html di folder docs
        with open(os.path.join(static_folder, "index.html"), "r") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")


# Endpoint to classify image
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty file name")

    try:
        # Read file bytes
        img_bytes = await file.read()
        img = load_img(io.BytesIO(img_bytes), target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Convert image array to PyTorch tensor
        img_tensor = torch.tensor(img_array).permute(0, 3, 1, 2).float()

        # Predict using the model
        with torch.no_grad():
            predictions = model(img_tensor)  # Get model predictions
            probabilities = torch.softmax(predictions[0], dim=0).numpy()  # Softmax to get probabilities

        # Sort results
        sorted_indices = np.argsort(probabilities)[::-1]
        results = [
            {'label': class_labels[i], 'probability': float(probabilities[i] * 100)}
            for i in sorted_indices
        ]

        return JSONResponse(content={'results': results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mengonfigurasi FastAPI untuk melayani file statis dari folder docs
# Pastikan path ke folder docs benar
static_folder = os.path.join(project_root, "final-deep-learning", "docs")
print(f"Calculated static folder path: {static_folder}")

# Mount folder docs untuk file statis
app.mount("/static", StaticFiles(directory=static_folder), name="static")


# Vercel Handler
handler = app

# Add API documentation
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    openapi_schema = get_openapi(
        title="Image Classification API",
        version="1.0",
        routes=app.routes,
    )
    openapi_schema["info"]["description"] = "API for image classification"
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=openapi_schema["info"]["title"],
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3.29.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3.29.0/swagger-ui.css",
    )

@app.get("/openapi.json", include_in_schema=False)
async def openapi():
    return get_openapi(
        title="Image Classification API",
        version="1.0",
        routes=app.routes,
    )

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")  # Default host untuk server
    port = int(os.getenv("PORT", 8000))  # Default port 8000
    uvicorn.run(app, host=host, port=port)
