from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 20

class TextRequest(BaseModel):
    text: str
    next_words: int

@app.get("/")
def root():
    return {"message": "Smart Text Predictor API is running"}

@app.post("/predict")
async def predict(request: TextRequest):
    seed_text = request.text
    for _ in range(request.next_words):
        tokens = tokenizer.texts_to_sequences([seed_text])[0]
        tokens = np.array([tokens[-max_len:]])
        tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=max_len)
        pred_index = np.argmax(model.predict(tokens, verbose=0))
        word = tokenizer.index_word.get(pred_index, "")
        seed_text += " " + word
    return {"prediction": seed_text}
