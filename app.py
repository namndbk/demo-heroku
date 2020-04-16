from flask import Flask, request, jsonify, render_template, request
import utils
import numpy as np
import pickle
from simple_naive_bayes import MultinomialNB

app = Flask(__name__)


DICT = utils.load_dict()
model = pickle.load(open("model.pkl", "rb"))


# model = pickle.load(open("addtone.pkl", "rb"))



@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
	if request.method == "GET":
		return render_template("index.html")
	else:
		document = request.form["document"]
		try:
			doc = utils.preprocess(document)
			doc = utils.bag_of_word(doc, DICT)
			y_pred = model.predict([doc])
			if int(y_pred[-1]) == 0:
				label = "positive"
			else:
				label = "negative"
		except:
			label = "Error !"
		return render_template("index.html", message=label, document=document)


if __name__ == '__main__':
	app.run()