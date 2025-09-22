from fastapi import FastAPI, Request, UploadFile, File
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load Keras model
model = tf.keras.models.load_model("hightide_model.keras")

# Prediction using JSON features
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    features = np.array(data.get("features", []), dtype=np.float32).reshape(1, -1)
    pred = model.predict(features)
    return {"alert_type": "high_tide", "severity": float(pred[0][0])}

# Optional: Prediction using uploaded file (if you plan to send files)
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    # Example: if CSV file with features
    content = await file.read()
    features = np.loadtxt(content.decode().splitlines(), delimiter=",")
    features = features.reshape(1, -1)
    pred = model.predict(features)
    return {"alert_type": "high_tide", "severity": float(pred[0][0])}

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}
