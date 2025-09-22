from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()

# Load model
model = tf.keras.models.load_model("hightide_model.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(file.file).resize((224, 224))  # adjust size as per model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run prediction
    pred = model.predict(img_array).tolist()
    
    return {"alert_type": "high_tide", "prediction": pred}

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

