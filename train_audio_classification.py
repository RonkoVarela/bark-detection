#!/usr/bin/env python

from datasets import load_dataset
import evaluate
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2CTCTokenizer
from transformers import TrainingArguments, Trainer
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2ForCTC


import torch
import numpy as np


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATASET_REPO = "rmarcosg/bark-detection"
BASELINE_MODEL = "facebook/wav2vec2-base-960h"
OUT_MODEL_REPO_ID = "rmarcosg/bark-detection-model"
METRICS_DICT = {
    metric: evaluate.load(metric, trust_remote_code=True)
    for metric in ["accuracy", "precision", "recall", "f1"]
}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        metric: function.compute(predictions=predictions, references=labels)
        for metric, function in METRICS_DICT.items()
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (positives weight 10 times one negative)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.], device=logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    print("load dataset")
    data = load_dataset(DATASET_REPO)

    print("load model")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(BASELINE_MODEL, num_labels=2).to(DEVICE)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASELINE_MODEL)

    # pre-encode dataset
    def prepare_sample(sample):
        # resample from 44.1 kHz to 16 kHz (44.1 kHz / 3 ~ 16 kHz)
        audio = sample["audio"]["array"][::3]
        return {'input_values': feature_extractor(audio, sampling_rate=16000).input_values[0]}

    encoded_data = data.map(prepare_sample, batched=False)
    train_dataset = encoded_data["train"]
    validation_dataset = encoded_data["validation"]
    test_dataset = test_data["test"]

    training_args = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        push_to_hub=True,
        hub_model_id=OUT_MODEL_REPO_ID,
    )

    print("load trainer")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    print("train")
    trainer.train()

    # eval
    test_metric = evaluate.load("f1")
    task_evaluator = evaluate.evaluator("audio-classification")
    results = task_evaluator.compute(
        model_or_pipeline=trainer.model,
        data=test_dataset,
        metric=test_metric,
        input_column='audio',
        feature_extractor=feature_extractor,
        label_mapping={"LABEL_0": 0, "LABEL_1": 1},
    )
    print("eval:", results)

    print("push")
    kwargs = {
        "finetuned_from": BASELINE_MODEL,
        "tasks": "audio-classification",
        "dataset": DATASET_REPO,
    }

    trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
