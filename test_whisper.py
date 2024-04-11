#!/usr/bin/env python

from datasets import load_dataset
from transformers import pipeline
from transformers import Trainer
from transformers import AutoModelForAudioClassification
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import torch
import click


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DATASET_REPO = "rmarcosg/bark-detection"
# BASELINE_MODEL = "openai/whisper-tiny.en"


@click.command()
@click.option('--baseline', default="openai/whisper-tiny.en", help="Model to use for prediction")
def main(baseline):
    """Main Function

    Example:
    python test_whisper.py [OPTIONS]

    Options:
    - --baseline: model to use to predict, default "openai/whisper-tiny.en"
    """
    print("load data")
    data = load_dataset(DATASET_REPO, split="test")
    print("load model")
    model = pipeline(
        "automatic-speech-recognition",
        model=baseline,
        device=DEVICE)
    sample = data[0]["audio"]
    print("file", data[0]["file"])
    print("predict")
    prediction = model(sample)["text"]
    print("prediction:", prediction)


if __name__ == '__main__':
    main()
