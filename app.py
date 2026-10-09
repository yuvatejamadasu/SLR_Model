import flask

from flask import Flask, render_template, request, app
import sklearn
from sklearn.linear_model import LinearRegression
import pickle
import numpy

with open("SLR_Model.pkl","rb") as f:
  model=pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def prediction():
    value = [float(i) for i in request.form.values()] #15
    value = numpy.array([value]) #15 --> numpy -->1D
    sol = model.predict(value)[0]
    return render_template("index.html", prediction_text=sol)

if __name__ == "__main__":
    app.run(debug=True)