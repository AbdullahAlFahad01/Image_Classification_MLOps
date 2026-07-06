from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# ==============================
# Create FastAPI App
# ==============================

app = FastAPI(
    title="Medicine Image Classifier",
    version="1.0"
)

# ==============================
# HTML Templates
# ==============================

templates = Jinja2Templates(directory="templates")

# ==============================
# Static Folder (CSS & JS)
# ==============================

app.mount("/static", StaticFiles(directory="static"), name="static")

# ==============================
# Load Model
# ==============================

model = tf.keras.models.load_model(
    "image_classifier.keras"
)

# ==============================
# Class Names
# ==============================

CLASS_NAMES = [
    "Afun",
    "Gelora",
    "Napa",
    "Revive"
]

IMAGE_SIZE = 224

# ==============================
# Home Page
# ==============================
from fastapi import Request

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
# ==============================
# Prediction API
# ==============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # run your CNN
   

    image = Image.open(
        BytesIO(await file.read())
    ).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    img_array = np.array(
        image,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        prediction[0][predicted_index]
    )

    return {

        "Predicted Class": predicted_class,

        "Confidence": round(
            confidence*100,
            2
        )

    }


