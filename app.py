import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
labelencoder= pickle.load(open('labelencoder_model.pkl', 'rb'))
regressor=pickle.load(open('dt_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = np.array(features)
    print(final_features)

    final_features[0]=labelencoder.transform([final_features[0]])[0]
    print(final_features)
    final_features=[float(x) for x in final_features]
    prediction = regressor.predict([final_features])[0]
    print(prediction)
    
    return render_template('index.html',prediction_text='The Cost Per Click of your AD: {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)