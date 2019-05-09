#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:43:57 2019

@author: jrscelza
"""

from flask import Flask
from flask import request
from sklearn.externals import joblib
import numpy as np

#lc ,lm, edo, tp, age, bmi

app = Flask(__name__)
 
@app.route("/compute")
def compute():
    
    lc = request.args.get('lc', default = 29, type = int)
    lm = request.args.get('lm', default = 5, type = int)
    edo = request.args.get('edo', default = 16, type = int)
    tp = request.args.get('tp', default = 2, type = int)
    age = request.args.get('age', default = 32, type = int)
    bmi = request.args.get('bmi', default = 24.8, type = float)
 
    
    pp = joblib.load('pre_processing_model.pkl')
    clf = joblib.load('classifier_model.pkl')
    
    data = np.array([lc,lm,edo,tp,age,bmi])
    
    pp_data = pp.transform(data.reshape(1,-1))
    
    prediction = clf.predict(pp_data)
    
    return str(prediction)
    
if __name__ == "__main__":
    app.run()


#http://127.0.0.1:5000/compute?lc=22&lm=3&edo=12&tp=2&age=20&bmi=27.0