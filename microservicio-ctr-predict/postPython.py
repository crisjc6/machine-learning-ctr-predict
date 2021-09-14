import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request, json
from sklearn import preprocessing


def abrirModelo(ruta):
    modelo = pickle.load(open(ruta, 'rb'))
    return modelo


def predecirXGB(X):
    params = {
    "objective": "binary:logistic",
    "booster" : "gbtree",
    "eval_metric": "logloss",
    "eta":0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "silent": 1,
    }
    modelo = xgb.XGBClassifier(**params)
    modelo.load_model("model_XGB.txt")
    return modelo.predict_proba(X)


app = Flask(__name__)

@app.route('/modeloXGB', methods=['POST'])
def modeloXGB():
    content = request.get_json()

    hour = content['hour']
    C1 = content['C1']
    banner_pos = content['banner_pos']
    site_id = content['site_id']
    site_domain = content['site_domain']
    site_category = content['site_category']
    app_id = content['app_id']
    app_domain = content['app_domain']
    app_category = content['app_category']
    device_id = content['device_id']
    device_ip = content['device_ip']
    device_model = content['device_model']
    device_type = content['device_type']
    device_conn_type = content['device_conn_type']
    C14 = content['C14']
    C15= content['C15']
    C16 = content['C16']
    C17 = content['C17']
    C18 = content['C18']
    C19 = content['C19']
    C20 = content['C20']
    C21= content['C21']

    datos = np.array([hour, C1, banner_pos, site_id, site_domain, site_category, app_id, app_domain,
                       app_category, device_id, device_ip, device_model, device_type, device_conn_type,
                       C14, C15, C16, C17, C18, C19, C20, C21]).reshape(1,-1)

    #scaler = pickle.load(open("scaler.dat", 'rb'))
    #datos = scaler.transform(datos)

    result = str(predecirXGB(datos)[0])

    d = {"result": result}

    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )

    return response



if __name__ == "__main__":
    app.run(threaded=False)
