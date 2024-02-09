from typing import Union
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/foo/")
def read_pkl(sepalLength: float, sepalWidth: float, petalLength: float, petalWidth: float):
    model = joblib.load('app/tree_iris_test.pkl')
    arguments = np.array([[sepalLength, sepalWidth, petalLength, petalWidth]])
    result = model.predict(arguments)
    return {"result": result[0]}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
