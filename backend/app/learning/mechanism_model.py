"""A small, inspectable classifier: which mechanism is a proposed plan likely to fail with?

Deliberately not a deep model. With the record counts this project currently has (18 saved runs, 13 of them
failures) anything larger would memorise and be reported as a capability. Multinomial logistic regression on
standardised features can be written down, audited, and *rejected* on evidence - which is what the training
script does when the gain over a majority baseline is not positive or the sample is too small to mean
anything.

The classifier is a candidate generator, never an authority: a prediction is a hypothesis with a stated
probability, and the physical verifier still decides what happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.app.learning.mechanism_dataset import FEATURES

#: Below this many examples a held-out score is noise, and the training report says NOT_ADOPTED.
MIN_EXAMPLES_FOR_ADOPTION = 40


def _matrix(examples):
    return np.array([[float(example['features'][name]) for name in FEATURES] for example in examples])


class MechanismModel:
    def __init__(self, classes, mean, scale, weights, bias, metadata=None):
        self.classes = list(classes)
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)
        self.weights = np.asarray(weights, float)
        self.bias = np.asarray(bias, float)
        self.metadata = dict(metadata or {})

    # -- training ---------------------------------------------------------------

    @classmethod
    def fit(cls, examples, *, epochs=600, rate=0.35, l2=1e-3, metadata=None):
        if not examples:
            raise ValueError('no examples')
        X = _matrix(examples)
        labels = [example['label'] for example in examples]
        classes = sorted(set(labels))
        if len(classes) < 2:
            raise ValueError(f'need at least two classes to train, got {classes}')
        index = {name: position for position, name in enumerate(classes)}
        Y = np.zeros((len(labels), len(classes)))
        Y[np.arange(len(labels)), [index[label] for label in labels]] = 1.
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0.] = 1.
        Z = (X - mean) / scale
        W = np.zeros((Z.shape[1], len(classes)))
        b = np.zeros(len(classes))
        for _ in range(epochs):
            logits = Z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            gradient = (P - Y) / len(labels)
            W -= rate * (Z.T @ gradient + l2 * W)
            b -= rate * gradient.sum(axis=0)
        return cls(classes, mean, scale, W, b, metadata)

    # -- inference --------------------------------------------------------------

    def predict_proba(self, features: dict) -> dict:
        x = np.array([float(features.get(name, 0.)) for name in FEATURES])
        z = (x - self.mean) / self.scale
        logits = z @ self.weights + self.bias
        logits = logits - logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return {name: float(value) for name, value in zip(self.classes, p)}

    def predict(self, features: dict) -> tuple[str, float]:
        probabilities = self.predict_proba(features)
        label = max(probabilities, key=probabilities.get)
        return label, probabilities[label]

    # -- persistence ------------------------------------------------------------

    def save(self, path):
        Path(path).write_text(json.dumps({
            'classes': self.classes, 'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
            'weights': self.weights.tolist(), 'bias': self.bias.tolist(),
            'features': list(FEATURES), 'metadata': self.metadata}, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if list(data.get('features') or []) != list(FEATURES):
            raise ValueError('model was trained on a different feature set')
        return cls(data['classes'], data['mean'], data['scale'], data['weights'], data['bias'],
                   data.get('metadata'))


def leave_one_out(examples) -> dict:
    """Score a model that never saw the example it is judged on, against the majority baseline."""
    labels = [example['label'] for example in examples]
    predictions, correct = [], 0
    for held_out in range(len(examples)):
        train = [example for index, example in enumerate(examples) if index != held_out]
        try:
            model = MechanismModel.fit(train)
        except ValueError:
            predictions.append(None)
            continue
        label, probability = model.predict(examples[held_out]['features'])
        predictions.append({'predicted': label, 'probability': probability,
                            'actual': examples[held_out]['label']})
        correct += int(label == examples[held_out]['label'])
    majority = max(set(labels), key=labels.count) if labels else None
    baseline = sum(1 for label in labels if label == majority)
    scored = [item for item in predictions if item]
    return {'examples': len(examples), 'scored': len(scored), 'correct': correct,
            'accuracy': (correct / len(scored)) if scored else None,
            'majority_class': majority,
            'baseline_accuracy': (baseline / len(labels)) if labels else None,
            'class_counts': {name: labels.count(name) for name in sorted(set(labels))},
            'predictions': predictions}
