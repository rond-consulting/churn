"""
Trying survival analysis for churn

Excellent blog: https://towardsdatascience.com/churn-prediction-and-prevention-in-python-2d454e5fd9a5
--> let's try to replicate that!

This is where you can find the data: https://www.kaggle.com/bandiatindra/telecom-churn-prediction

Lifelines API documentation: https://lifelines.readthedocs.io/en/latest/References.html

More simple blog, but not bad either: https://towardsdatascience.com/survival-analysis-to-understand-customer-retention-e3724f3f7ea2

Paper on survival analysis for churn - with badly formatted formulas: http://ceur-ws.org/Vol-2577/paper5.pdf

Author: Hans Weda, rond consulting
Date: 11 february 2021

"""

import os

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

from src.visualization.prediction_plot_utils import plot_result

load_dotenv(find_dotenv())

from kaggle.api.kaggle_api_extended import KaggleApi


DATA_DIR = os.path.join("..", "data", "raw")


def load_from_kaggle(owner: str, dataset_name: str) -> pd.DataFrame:
    """
    This function downloads data from kaggle into DATA_DIR and returns the first file (alphabetically).
    Following directions on https://medium.com/analytics-vidhya/fetch-data-from-kaggle-with-python-9154a4c610e3
    The documentation of the kaggle api can be found here: https://github.com/Kaggle/kaggle-api/tree/master/kaggle
    :param owner: the owner of the dataset on Kaggle
    :param dataset_name: the name of the dataset
    :return: the alphabetically first file from the downloaded dataset
    """
    # make sure raw data directory exists
    target_directory = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(target_directory, exist_ok=True)

    # connect to kaggle
    api = KaggleApi()
    api.authenticate()

    # download if it doesn't exist already and unzip
    api.dataset_download_files(dataset=f"{owner}/{dataset_name}", path=target_directory, unzip=True)

    # read the first file and return
    df = pd.read_csv(os.path.join(target_directory, os.listdir(target_directory)[0]))

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(
        df[[
            'gender',
            'SeniorCitizen',
            'Partner',
            'Dependents',
            'tenure',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod',
            'Churn'
        ]], drop_first=True
    )
    df = dummies.join(df[['MonthlyCharges', 'TotalCharges']])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


if __name__ == "__main__":
    # loading from kaggle
    df = load_from_kaggle(owner="blastchar", dataset_name="telco-customer-churn")
    # clean the data
    df = clean_data(df)
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        df.dropna().drop(columns=["Churn_Yes"]),
        df.dropna()['Churn_Yes']
    )
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)

    """
    fig = plot_result(
        model=clf,
        X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train,
        model_name="logistic regression"
    )
#    confusion_matrix(y_pred=clf.predict(X_test), y_true=y_test)
    plot_confusion_matrix(clf, X=X_test, y_true=y_test, cmap=plt.cm.Blues)
    plt.show()
    """

    cph = CoxPHFitter()
    cph_train, cph_test = train_test_split(df.dropna(), test_size=0.2)
    cph.fit(cph_train, 'tenure', 'Churn_Yes')

    print("finished!")
