from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load the pre-trained Hugging Face model (in this case, for sentiment analysis)
model = pipeline("sentiment-analysis", model='./model')

# Define the input schema
class InputData(BaseModel):
    text: str

# API endpoint to get predictions
@app.post("/predict")
async def predict(data: InputData):
    # Use the model to get predictions
    prediction = model(data.text)
    return {"prediction": prediction}
