import logging
import azure.functions as func
import json
import joblib
import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .logregmodel import apply_model

def main(req: func.HttpRequest, inputBlob: func.InputStream, outputQueue: func.Out[func.QueueMessage]) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    url = req.params.get('url')
    if not url:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            url = req_body.get('url')

    if url:     
        names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        dataframe = pd.read_csv(url, names=names)     
        xscaled = apply_model(dataframe)
        model = pickle.loads(inputBlob.read())
        y_pred = model.predict(xscaled)
        y_pred_str = str(y_pred)
        return func.HttpResponse(y_pred_str)
        outputQueue.set(url)

    else:
        return func.HttpResponse(
             "Please pass a name on the query string or in the request body",
             status_code=400
        )
