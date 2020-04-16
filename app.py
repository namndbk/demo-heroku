from flask import Flask
import pickle
from flask import render_template, request, redirect
import preprocess
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np



app = Flask(__name__)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
	if request.method == "GET":
		return render_template("index.html")
	else:
		document = request.form["document"]
		document = document.strip()
		inp = []
		inp.append(preprocess.encode(document))
		inp = np.array(inp)
		y_pred = model.predict(inp)
		out = preprocess.decode(y_pred[-1])
		return render_template("index.html", message=out, document=document)

if __name__ == '__main__':
	app.run()