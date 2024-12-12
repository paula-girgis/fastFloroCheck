from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
from pydantic import BaseModel
import google.generativeai as genai
import io

# Configure the Generative AI model
genai.configure(api_key="AIzaSyCNoCBpW21-5V7SPgJ1duIBKjYSxybbeM4")
generative_model = genai.GenerativeModel("gemini-2.0-flash-exp")



# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the TensorFlow model once during app startup
try:
    model = tf.keras.models.load_model("best_model.keras")
    # Pre-run a dummy input to optimize the graph
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy_input)
except Exception:
    model = None

# Disease class mapping
class_map = {
    0: 'Apple Cedar Rust',
    1: 'Apple Healthy',
    2: 'Apple Scab',
    3: 'Bluberry Healthy',
    4: 'Citrus Black Spot',
    5: 'Citrus Canker',
    6: 'Citrus Greening',
    7: 'Citrus Healthy',
    8: 'Corn Gray Leaf Spot',
    9: 'Corn Northern Leaf Blight',
    10: 'Grape Healthy',
    11: 'Pepper, Bell Bacterial Spot',
    12: 'Pepper, Bell Healthy',
    13: 'Potato Early Blight',
    14: 'Potato Healthy',
    15: 'Potato Late Blight',
    16: 'Raspberry Healthy',
    17: 'Strawberry Healthy',
    18: 'Strawberry Leaf Scorch',
    19: 'Tomato Early Blight',
    20: 'Tomato Healthy',
    21: 'Tomato Late Blight',
    22: 'Tomato Yellow Leaf Curl Virus'
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def preprocess_image(file: bytes) -> np.ndarray:
    """Prepare image for prediction."""
    try:
        img = Image.open(io.BytesIO(file)).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

def calculate_green_percentage(img: Image.Image) -> float:
    """Calculate the percentage of green pixels in an image."""
    img_array = np.array(img)
    green_mask = (
        (img_array[:, :, 1] > img_array[:, :, 0]) &  # Green channel > Red channel
        (img_array[:, :, 1] > img_array[:, :, 2]) &  # Green channel > Blue channel
        (img_array[:, :, 1] > 50)                   # Green channel > Threshold
    )
    green_percentage = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
    return green_percentage * 100

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from an uploaded image."""
    # Validate file extension
    if file.filename.rsplit('.', 1)[-1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: PNG, JPG, JPEG.")

    # Read and preprocess the image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    # Filter based on green color percentage
    green_percentage = calculate_green_percentage(img)
    if green_percentage < 9.19:
        raise HTTPException(status_code=400, detail="Unclear image. Please upload a clear plant leaf image")

    img_array = preprocess_image(content)

    # Ensure the model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Predict and interpret the results
    predictions = model.predict(img_array, verbose=0)
    confidence = float(np.max(predictions[0]))
    predicted_class_idx = int(np.argmax(predictions[0]))

    # Handle low-confidence predictions
    if confidence < 0.53:
        raise HTTPException(status_code=400, detail="Unclear image. Please upload a clear plant leaf image")

    predicted_class = class_map.get(predicted_class_idx, "Disease not supported")
    return {"predicted_class": predicted_class}

@app.get("/list_models")
async def list_models():
    try:
        models = genai.list_models()
        return {"available_models": [model.name for model in models]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

# Define a Pydantic model for the request body
class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chatbot(request: ChatRequest):
    user_input = request.user_input.strip()  # Access the input from the request body
    if not user_input:
        raise HTTPException(status_code=400, detail="Input or image is required.")

    try:
        # Process image if available
        # Generate response using Generative AI
        response = generative_model.generate_content(user_input)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
    
# Catch-all route for undefined endpoints
@app.api_route("/{path_name:path}")
async def catch_all_routes(path_name: str):
    """Handle undefined endpoints."""
    raise HTTPException(status_code=404, detail="Endpoint not found.")
