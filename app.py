from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the model with error handling
try:
    model = tf.keras.models.load_model("best_model.keras")
    logging.info("Model loaded successfully!")
except (OSError, ValueError) as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define the disease classes
class_indices = { ... }  # Keep your class mapping
class_map = {value: key for key, value in class_indices.items()}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    """Check if the file has a valid extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file: bytes) -> np.ndarray:
    """Preprocess the image file and prepare it for prediction"""
    try:
        img = Image.open(io.BytesIO(file))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

@app.get("/")
def root():
    return {"message": "Welcome to the Plant Disease Prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint"""
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        img_array = preprocess_image(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")

    try:
        prediction = model.predict(img_array)
        confidence_scores = prediction[0]
        max_confidence = float(np.max(confidence_scores))
        predicted_class_idx = np.argmax(confidence_scores)

        threshold = 0.4
        if max_confidence < threshold:
            raise HTTPException(status_code=400, detail="Unclear image. Please upload a clear plant leaf image.")

        if predicted_class_idx in class_map:
            predicted_class = class_map[predicted_class_idx]
            return JSONResponse({"predicted_class": predicted_class})
        else:
            raise HTTPException(status_code=400, detail="Disease not supported yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all_routes(path_name: str):
    logging.warning(f"Unhandled route accessed: {path_name}")
    raise HTTPException(
        status_code=404,
        detail=f"The endpoint '/{path_name}' does not exist. Please check the URL."
    )
