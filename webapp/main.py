from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import librosa


model = pipeline('audio-classification',
                 model='rmarcosg/bark-detection',
                 feature_extractor='rmarcosg/bark-detection',
                )

app = FastAPI()


@app.get('/')
def root():
    return HTMLResponse("<h1>A self-documenting API to classify audio.</h1>")


@app.post("/classify/")
async def classify_audio(file: UploadFile = File(...)):
    audio_input, sr = librosa.load(file.file, sr=None)
    if sr != 16000:
      audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)

    result = model(resampled_audio)
    return {"bark probability": result[0]['score']}
