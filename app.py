from fastapi import FastAPI
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn

app = FastAPI()

model = load_model("text_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 20

@app.get("/")
def home():
    return {"message": "Smart Text Predictor API is running"}

@app.post("/predict")
def predict(text: str, next_words: int = 3):
    seed_text = text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list[-max_len:]], maxlen=max_len)
        predicted_index = np.argmax(model.predict(token_list, verbose=0))
        seed_text += " " + tokenizer.index_word.get(predicted_index, "")
    return {"prediction": seed_text}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
