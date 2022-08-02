#importing libraries
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('E:\Customer Segmentation Project\ML model\clustermodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('HomeScreen.html')


if __name__=="__main__":
    app.run(debug=True)