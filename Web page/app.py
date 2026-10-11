from pathlib import Path

import pickle

from flask import Flask, request, render_template
import numpy as np

from mental_health import MODEL_FILE, sigmoid, train_model

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR), static_folder=str(BASE_DIR), static_url_path='/static')


def load_model():
    if not MODEL_FILE.exists():
        train_model()

    with MODEL_FILE.open('rb') as model_file:
        return pickle.load(model_file)


model = load_model()


def parse_int_field(form, name, minimum, maximum):
    raw_value = form.get(name, '').strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f'{name.replace("_", " ")} must be a number.') from exc

    if not minimum <= value <= maximum:
        raise ValueError(f'{name.replace("_", " ")} must be between {minimum} and {maximum}.')
    return value


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = np.asarray([[
            parse_int_field(request.form, 'Age', 15, 100),
            parse_int_field(request.form, 'Gender', 0, 2),
            parse_int_field(request.form, 'Family_history', 0, 1),
        ]], dtype=float)
    except ValueError as exc:
        return render_template('index.html', error=str(exc), values=request.form), 400

    scaled = (features - model['mean']) / model['scale']
    probability = float(sigmoid(scaled @ model['weights'] + model['intercept'])[0])
    output = f'{probability:.2f}'

    if probability > 0.5:
        message = f'You may need treatment. Probability of mental illness is {output}.'
    else:
        message = f'You may not need treatment. Probability of mental illness is {output}.'
    return render_template('index.html', pred=message, values=request.form)


if __name__ == '__main__':
    app.run(debug=True)
