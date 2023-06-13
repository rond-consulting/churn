# tryout api

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

# ML model try-out
#url = "http://localhost:8000/model_predict"
#url = "http://localhost:80/model_predict"
url = "https://purple-api-container.azurewebsites.net/model_predict"

# auth = HTTPBasicAuth("access_token", "bla")
headers = {'access_token': 'secret'}

df = pd.DataFrame({
    "fin": [0, 0, 0, 1, 0],
    "age": [27, 18, 19, 23, 19],
    "wexp": [0, 0, 1, 1, 1],
    "prio": [3, 8, 13, 1, 3]
})

response = requests.post(
    url,
    json=df.to_dict(orient="records"),
    #params={"percentile": 1.0},
    headers=headers
)
print("local:")
print(response.text)
