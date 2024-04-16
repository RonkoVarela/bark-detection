from transformers import pipeline
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np


model = pipeline('audio-classification',
                 model='rmarcosg/bark-detection',
                 feature_extractor='rmarcosg/bark-detection',
                )

app = FastAPI()


class Body(BaseModel):
    audio: np.array


@app.get('/')
def root():
    return HTMLResponse("<h1>A self-documenting API to classify audio.</h1>")


@app.post('/classify')
def predict(body: Body):
    results = model(body.audio)
    return np.argmax(results, axis=1)
