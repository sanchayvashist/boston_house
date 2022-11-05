import numpy as np
import pandas as pd
import pickle 
from flask import Flask ,app ,request ,jsonify ,render_template ,url_for
app=Flask(__name__)
reg_model=pickle.load(open('regm.pkl','rb'))        #load regression model
scaler=pickle.load(open('scale.pkl','rb'))          #load scaling model
@app.route('/')
def home():
    return render_template('home.html')             #redirect to home.html
@app.route('/predict_api',methods=['POST'])         #api with post request becuase we take input then generate output using it
def predict():
    data=request.json['data']
    print(data)                                         #data is dictionory with elements in key value pair
    print(np.array(list(data.values())).reshape(1,-1))  #convert all inputs in a single file of data to add in prediction model
    transformed_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=reg_model.predict(transformed_data)
    print(output[0])                                       #output recived is an 2d array
    return jsonify(output[0])                               #return jsonifoed output
if __name__=="__main__":
    app.run(debug=True)
