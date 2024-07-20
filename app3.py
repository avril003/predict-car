import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='static')

model = pickle.load(open('model_avril.pkl', 'rb'))

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/Charts")
def Charts():
    return render_template('linechart.html')

@app.route("/Form")
def form():
    return render_template('form.html')

@app.route('/predict', methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)

    return render_template("form.html", prediction_text = "Hasil Prediksi harga mobil {}".format(prediction[0]))

if __name__ =="__main__":
    app.run(debug=True)
