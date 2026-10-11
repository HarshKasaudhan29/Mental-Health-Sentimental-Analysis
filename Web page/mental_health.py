"""Train the web application's treatment-prediction model without scikit-learn."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / 'survey.csv'
MODEL_FILE = BASE_DIR / 'model.pkl'

data = pd.read_csv(DATA_FILE)
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = data[data['Age'].between(15, 100)].copy()

gender = data['Gender'].fillna('').str.lower().str.strip()
data['gender_code'] = 2
data.loc[gender.str.contains('female|woman', regex=True), 'gender_code'] = 1
data.loc[gender.str.contains('male|man', regex=True), 'gender_code'] = 0
data['family_history_code'] = data['family_history'].map({'No': 0, 'Yes': 1})
data['treatment_code'] = data['treatment'].map({'No': 0, 'Yes': 1})
data = data.dropna(subset=['family_history_code', 'treatment_code'])

features = data[['Age', 'gender_code', 'family_history_code']].to_numpy(dtype=float)
target = data['treatment_code'].to_numpy(dtype=float)
mean = features.mean(axis=0)
scale = features.std(axis=0)
scale[scale == 0] = 1
features = (features - mean) / scale

weights = np.zeros(features.shape[1])
intercept = 0.0
for _ in range(2_000):
    probabilities = 1 / (1 + np.exp(-(features @ weights + intercept)))
    error = probabilities - target
    weights -= 0.1 * (features.T @ error / len(target))
    intercept -= 0.1 * error.mean()

with MODEL_FILE.open('wb') as model_file:
    pickle.dump({'mean': mean, 'scale': scale, 'weights': weights, 'intercept': intercept}, model_file)

print(f'Trained model saved to {MODEL_FILE}')
