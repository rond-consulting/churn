"""
Try-out of an API to deploy a ML survival model.

The API is built using FastAPI. It uses a model of the data, or - more precisely - of a data
record. This is defined in the class auto(BaseModel). Using such a model ensures that the right
data-type is used - it throws an error if this is not the case. The uploaded dataframe may have
many other columns, but should at least have those that are defined in class auto(BaseModel).

Sources:
https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html


Author: Hans Weda, @ rond consulting
Date: February 6th, 2023
"""

import pickle
from typing import List

import pandas as pd
from fastapi import FastAPI, Query, Depends
from fastapi.responses import RedirectResponse
from fastapi.security.api_key import APIKey
from pydantic import BaseModel

import auth

app = FastAPI(
    title="Tryout survival ML api",
    description=(
        "Trying out an api to deploy a simple survival regression model on the Rossi dataset. "
        "Hans Weda @ rond consulting"
    )
)

filename = 'survival_model.pkl'


class boef(BaseModel):
    fin: int
    wexp: int
    age: int
    prio: int


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.post("/model_predict")
async def get_predictions(
        boeven: List[boef],
        percentile: float = Query(title="prediction", default=0.8, ge=0, le=1),
        api_key: APIKey = Depends(auth.get_api_key)
) -> List[float]:
    """
    Return predictions for the rossi dataset.

    :param boeven: list of properly formatted dictionaries
    :param percentile: the percentile, must be between 0 and 1
    :param api_key: the Api-Key
    :return: list with predictions
    """

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    # create data-frame from load
    df = pd.DataFrame([a.dict() for a in boeven])

    # calculate predictions
    preds = loaded_model.predict_percentile(df, p=percentile)

    # make sure preds is always a pandas Series
    preds = pd.Series(preds)

    # replace infinity
    with pd.option_context('mode.use_inf_as_na', True):
        preds = preds.fillna(-999)
    # print("in main")
    # print(api_key)
    # print(preds)  # for debugging purposes
    # print("probability: {}".format(percentile))

    return preds.tolist()
