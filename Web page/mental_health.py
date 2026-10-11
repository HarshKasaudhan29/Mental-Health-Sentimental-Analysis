"""Train the web application's treatment-prediction model without scikit-learn."""

from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / 'survey.csv'
MODEL_FILE = BASE_DIR / 'model.pkl'


def normalize_gender(value):
    """Return 0 for male, 1 for female, and 2 for other/unknown."""
    text = str(value).lower().strip()
    words = set(re.findall(r'[a-z]+', text))

    female_terms = {'f', 'female', 'woman', 'cisfemale'}
    male_terms = {'m', 'male', 'man', 'cismale'}

    if words & female_terms:
        return 1
    if words & male_terms:
        return 0
    return 2


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -500, 500)))


def train_model(data_file=DATA_FILE, model_file=MODEL_FILE):
    data = pd.read_csv(data_file)
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data = data[data['Age'].between(15, 100)].copy()

    data['gender_code'] = data['Gender'].apply(normalize_gender)
    data['family_history_code'] = data['family_history'].str.strip().str.lower().map({'no': 0, 'yes': 1})
    data['treatment_code'] = data['treatment'].str.strip().str.lower().map({'no': 0, 'yes': 1})
    data = data.dropna(subset=['Age', 'family_history_code', 'treatment_code'])

    features = data[['Age', 'gender_code', 'family_history_code']].to_numpy(dtype=float)
    target = data['treatment_code'].to_numpy(dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    features = (features - mean) / scale

    weights = np.zeros(features.shape[1])
    intercept = 0.0
    for _ in range(2_000):
        probabilities = sigmoid(features @ weights + intercept)
        error = probabilities - target
        weights -= 0.1 * (features.T @ error / len(target))
        intercept -= 0.1 * error.mean()

    model = {'mean': mean, 'scale': scale, 'weights': weights, 'intercept': intercept}
    with model_file.open('wb') as handle:
        pickle.dump(model, handle)
    return model_file


if __name__ == '__main__':
    saved_model = train_model()
    print(f'Trained model saved to {saved_model}')
