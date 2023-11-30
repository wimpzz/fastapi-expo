from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Form
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./model_keras/plants"

MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES = ['Aloe Vera',
                'Basil',
                'Cabbage Succulent',
                'Chinese Evergreen',
                'Golden Pothos',
                'Peace Lily',
                'Rubber Tree',
                'Snake Plant',
                'Spider Plant',
                'ZZ Plant']

class PredictionResult(BaseModel):
    class1: str
    confidence1: float

@app.get("/")
def root():
    return {"Welcome": "Hello World!!"}

@app.get("/test")
def ping():
    return {"Test": "FastAPI is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((256, 256))

    img_array = np.array(image)
    img_batch = np.expand_dims(img_array, 0)

    predictions = MODEL.predict(img_batch)

    top_classes = [CLASS_NAMES[idx] for idx in np.argsort(predictions[0])[-1:][::-1]]
    top_confidences = [float(conf) * 100 for conf in np.sort(predictions[0])[-1:][::-1]]

    result = PredictionResult(class1=top_classes[0], confidence1=top_confidences[0])
    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)