from typing import Union
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import requests

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/foo/")
def read_pkl(sepalLength: float, sepalWidth: float, petalLength: float, petalWidth: float):
    url = 'https://uai-nico.s3.amazonaws.com/tree_iris_test.pkl'
    save_path = 'model/tree_iris_test.pkl'
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        model = joblib.load('model/tree_iris_test.pkl')
        arguments = np.array([[sepalLength, sepalWidth, petalLength, petalWidth]])
        result = model.predict(arguments)
        
        print("File downloaded successfully")
    else:
        print("Failed to download file. Status code:", response.status_code)

    
    return {"result": result[0]}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
