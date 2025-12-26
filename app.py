from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

# Allow all origins (frontend on any domain can access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class PredictRequest(BaseModel):
    text: str
    next_words: int

# Load your trained TensorFlow/Keras model
model = tf.keras.models.load_model("model")  # make sure your model is in "model/" folder

@app.post("/predict")
async def predict(req: PredictRequest):
    input_text = req.text
    next_words = req.next_words
    
    # Example: replace this with actual model prediction logic
    # For demonstration, we just repeat the input text
    prediction = f"{input_text} ... predicted {next_words} words"
    
    return {"prediction": prediction}

@app.get("/")
def root():
    return {"message": "Smart Text Predictor API is running"}
