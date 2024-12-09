from fastapi import FastAPI, File, UploadFile, HTTPException, status
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
class_indices = {
    'Apple Cedar Rust': 0,
    'Apple Healthy': 1,
    'Apple Scab': 2,
    'Bluberry Healthy': 3,
    'Citrus Black Spot': 4,
    'Citrus Canker': 5,
    'Citrus Greening': 6,
    'Citrus Healthy': 7,
    'Corn Gray Leaf Spot': 8,
    'Corn Northern Leaf Blight': 9,
    'Grape Healthy': 10,
    'Pepper,bell Bacterial Spot': 11,
    'Pepper,bell Healthy': 12,
    'Potato Early Blight': 13,
    'Potato Healthy': 14,
    'Potato Late Blight': 15,
    'Raspberry Healthy': 16,
    'Strawberry Healthy': 17,
    'Strawberry Leaf Scorch': 18,
    'Tomato Early Blight': 19,
    'Tomato Healthy': 20,
    'Tomato Late Blight': 21,
    'Tomato Yellow Leaf Curl Virus': 22
}

class_map = {value: key for key, value in class_indices.items()}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file: bytes) -> np.ndarray:
    """Preprocess the image file and prepare it for prediction."""
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file."
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint."""
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        img_array = preprocess_image(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image preprocessing failed: {e}"
        )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded. Please try again later."
        )

    try:
        prediction = model.predict(img_array)
        confidence_scores = prediction[0]
        max_confidence = float(np.max(confidence_scores))
        predicted_class_idx = np.argmax(confidence_scores)

        threshold = 0.4
        if max_confidence < threshold:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unclear image. Please upload a clear plant leaf image."
            )

        if predicted_class_idx in class_map:
            predicted_class = class_map[predicted_class_idx]
            return JSONResponse({"predicted_class": predicted_class})
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Disease not supported yet."
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {e}"
        )

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all_routes(path_name: str):
    """Catch-all route for undefined endpoints."""
    logging.warning(f"Unhandled route accessed: {path_name}")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"The endpoint '/{path_name}' does not exist. Please check the URL."
    )
