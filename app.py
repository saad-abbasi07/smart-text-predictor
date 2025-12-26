from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Allow any website to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # <- this allows all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved model and tokenizer
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
        output_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + output_word
    return {"prediction": seed_text}
