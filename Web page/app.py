from pathlib import Path

from flask import Flask, request, render_template
import pickle
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR))

with (BASE_DIR / 'model.pkl').open('rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    features = np.asarray(final, dtype=float)
    scaled = (features - model['mean']) / model['scale']
    probability = 1 / (1 + np.exp(-(scaled @ model['weights'] + model['intercept'])))
    output = '{0:.{1}f}'.format(probability[0], 2)

    if float(output) > 0.5:
        return render_template('index.html',pred='You need a treatment.\nProbability of mental illness is {}'.format(output))
    else:
        return render_template('index.html',pred='You do not need treatment.\n Probability of mental illness is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
