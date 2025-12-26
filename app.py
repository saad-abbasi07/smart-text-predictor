import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = tf.keras.models.load_model("model.keras")

class TextRequest(BaseModel):
    text: str
    next_words: int

@app.post("/predict")
async def predict(request: TextRequest):
    input_text = request.text
    next_words = request.next_words
    prediction = input_text + " ..."  
    return {"prediction": prediction}
