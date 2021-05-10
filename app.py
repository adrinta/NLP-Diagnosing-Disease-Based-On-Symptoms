import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from nltk import PorterStemmer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
le = pickle.load(open('le.pkl', 'rb'))

ps = PorterStemmer()

def normalize(text):
    if text == text:
        text = text.lower()
        text = text.replace('_', ' ')
        text = ' '.join([ps.stem(word) for word in text.split()])
    else:
        text = ''
    return text

@app.route('/')
def home():
    return render_template('index.html', prediction_text='...')

@app.route('/predict',methods=['POST'])
def predict():

    val = [val for val in request.form.values()]
    text = val[2]
    text = normalize(text)
    text = vectorizer.transform([text])
    prediction = model.predict(text)
    prediction = le.inverse_transform(prediction)

    output = prediction[0]

    return render_template('index.html', prediction_text='{}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)