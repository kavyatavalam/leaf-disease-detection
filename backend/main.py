from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from auth import router as auth_router
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import pickle
import os

app = FastAPI()
Base.metadata.create_all(bind=engine)
# ✅ CORS
origins = [
    "http://localhost",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
# ✅ Load model (ONLY ONCE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model_1.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.pkl")

MODEL = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    CLASS_NAMES = pickle.load(f)

# ✅ Warm-up (optional but good)
MODEL.predict(np.zeros((1,128,128,3)))

# ✅ Image processing
def read_file_as_image(data):
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((128,128))
    image = np.array(image) / 255.0
    return image

# ✅ Routes
@app.get("/")
def home():
    return {"message": "API working"}

@app.get("/health")
def health():
    return {"status": "running"}

from models import Prediction
from database import SessionLocal

@app.post("/predict")
async def predict(file: UploadFile = File(...), username: str = None):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        # ✅ SAVE TO DATABASE
        if username:
            db = SessionLocal()
            new_entry = Prediction(
                username=username,
                disease=predicted_class,
                confidence=confidence
            )
            db.add(new_entry)
            db.commit()
            db.close()

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/history")
def get_history(username: str):
    db = SessionLocal()

    records = db.query(Prediction).filter(Prediction.username == username).all()

    db.close()

    return [
        {
            "disease": r.disease,
            "confidence": r.confidence
        }
        for r in records
    ]

# ✅ Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)