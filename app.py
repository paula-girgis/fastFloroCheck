from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

app = FastAPI()

# Enable CORS middleware (optional, but useful if integrating with frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend's origin if required
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the model with error handling
try:
    model = tf.keras.models.load_model(r"best_model.keras")
    print("Model loaded successfully!")
except (OSError, ValueError) as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None to prevent usage if not loaded

# Define the disease classes
class_indices = {
    'Apple Cedar Rust': 0,
    'Apple Healthy': 1,
    'Apple Scab': 2,
    'Bluberry Healthy': 3,
    'Citrus Black Spot': 4,
    'Citrus Canker': 5,
    'Citrus Healthy': 6,
    'Corn Gray Leaf Spot': 7,
    'Corn Northern Leaf Blight': 8,
    'Grape Healthy': 9,
    'Pepper,bell Bacterial Spot': 10,
    'Pepper,bell Healthy': 11,
    'Potato Early Blight': 12,
    'Potato Healthy': 13,
    'Potato Late Blight': 14,
    'Raspberry Healthy': 15,
    'Strawberry Healthy': 16,
    'Strawberry Leaf Scorch': 17,
    'Tomato Early Blight': 18,
    'Tomato Healthy': 19,
    'Tomato Late Blight': 20
}

class_map = {value: key for key, value in class_indices.items()}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    """Check if the file has a valid extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file: bytes) -> np.ndarray:
    """Preprocess the image file and prepare it for prediction"""
    img = Image.open(io.BytesIO(file))  # Use io.BytesIO to handle in-memory file
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint"""
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Read the file and preprocess
        content = await file.read()
        img_array = preprocess_image(content)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Ensure the file is a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    # Predict with the model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    prediction = model.predict(img_array)

    confidence_scores = prediction[0]
    max_confidence = float(np.max(confidence_scores))
    predicted_class_idx = np.argmax(confidence_scores)

    # Confidence threshold
    threshold = 0.68
    if max_confidence < threshold:
        raise HTTPException(status_code=400, detail="Invalid photo. Please upload a plant leaf image.")

    # Return the predicted class directly using class_map
    if predicted_class_idx in class_map:
        return JSONResponse({
            "predicted_class": class_map[predicted_class_idx]
        })
    else:
        raise HTTPException(status_code=400, detail="Disease not supported yet.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Detection API!"}
