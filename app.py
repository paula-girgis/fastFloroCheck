from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from aiocache import cached
import time

# Initialize FastAPI app
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
    model = tf.keras.models.load_model("best_model.keras")
    print("Model loaded successfully!")
except (OSError, ValueError) as e:
    print(f"Error loading model: {e}")
    model = None  # Ensure model is None if loading fails

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Efficient image preprocessing with OpenCV
def preprocess_image(file: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img, axis=0)
    return img_array

@app.post("/predict")
@cached(ttl=300)  # Cache the endpoint for 5 minutes
async def predict(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Read the file and preprocess
        content = await file.read()
        img_array = preprocess_image(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    # Predict with the model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    start_time = time.time()
    prediction = model.predict(img_array)
    response_time = round(time.time() - start_time, 2)

    confidence_scores = prediction[0]
    max_confidence = float(np.max(confidence_scores))
    predicted_class_idx = int(np.argmax(confidence_scores))

    # Confidence threshold
    threshold = 0.68
    if max_confidence < threshold:
        raise HTTPException(status_code=400, detail="Invalid photo. Please upload a plant leaf image.")

    # Map predicted class
    if predicted_class_idx in class_map:
        return JSONResponse({
            "predicted_class": class_map[predicted_class_idx],
            "confidence": max_confidence,
            "response_time": response_time
        })
    else:
        raise HTTPException(status_code=400, detail="Disease not supported yet.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Detection API!"}
