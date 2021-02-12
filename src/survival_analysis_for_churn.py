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
from lifelines.utils import median_survival_times, qth_survival_times
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split

from src.visualization.prediction_plot_utils import plot_result
from src.models.model_utils import remove_collinear_variables
from src.data.data_utils import load_from_kaggle


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
    # remove nans
    # ToDo: statistics on amount of nans, or something clever on replacing nans
    df = df.dropna()
    return df


if __name__ == "__main__":
    # loading from kaggle
    df = load_from_kaggle(owner="blastchar", dataset_name="telco-customer-churn")
    # clean the data
    df = clean_data(df)
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["Churn_Yes"]),
        df['Churn_Yes']
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
    df = remove_collinear_variables(df, target="Churn_Yes", thresh=100)

    cph = CoxPHFitter()
    cph_train, cph_test = train_test_split(df, test_size=0.2)
    cph.fit(cph_train, 'tenure', 'Churn_Yes')

    # result
    print(cph.print_summary())
    cph.plot()
    plt.tight_layout()
    plt.show()

    cph.plot_partial_effects_on_outcome('TotalCharges', values=[0, 4000])
    plt.show()

    censored_subjects = df.loc[df['Churn_Yes'] == 0]

    unconditioned_sf = cph.predict_survival_function(censored_subjects)

    conditioned_sf = unconditioned_sf.apply(lambda c: (c / c.loc[df.loc[c.name, 'tenure']]).clip(upper=1))

    subject = 12
    unconditioned_sf[subject].plot(ls="--", color="#A60628", label="unconditioned")
    conditioned_sf[subject].plot(color="#A60628", label="conditioned on $T>58$")
    plt.legend()

    predictions_50 = median_survival_times(conditioned_sf)
    # This is the same, but you can change the fraction to get other
    # %tiles.
    # predictions_50 = qth_survival_times(.50, conditioned_sf)

    values = predictions_50.T.join(df[['MonthlyCharges', 'tenure']])
    values['RemainingValue'] = values['MonthlyCharges'] * (values[0.5] - values['tenure'])

    # ToDo: you cannot have both contracts at the same time.
    #  Modify the code such that the contract is either one or two year.
    #  The same for payment method
    upgrades = [
        'PaymentMethod_Credit card (automatic)',
        'Contract_One year',
        'Contract_Two year'
    ]
    results_dict = {}
    for customer in values.index:
        actual = df.loc[[customer]]
        change = df.loc[[customer]]
        results_dict[customer] = [cph.predict_median(actual)]
        for upgrade in upgrades:
            change[upgrade] = 1 if list(change[upgrade]) == [0] else 0
            results_dict[customer].append(cph.predict_median(change))
            change[upgrade] = 1 if list(change[upgrade]) == [0] else 0
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = ['baseline'] + upgrades
    actions = values.join(results_df).drop([0.5], axis=1)

    loss_dict = {}
    for i in range(1, 73):
        score = brier_score_loss(
            y_true=cph_test['Churn_Yes'],
            y_prob=1 - cph.predict_survival_function(cph_test).loc[i].values,
            pos_label=1)
        loss_dict[i] = [score]
    loss_df = pd.DataFrame(loss_dict).T
    fig, ax = plt.subplots()
    loss_df.plot(ax=ax)
    ax.set(xlabel='Prediction Time', ylabel='Calibration Loss', title='Cox PH Model Calibration Loss / Time')
    ax.grid()
    plt.show()

    loss_df.columns = ['loss']
    temp_df = actions.reset_index().set_index('PaymentMethod_Credit card (automatic)').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['CreditCard Lower'] = temp_df['CreditCard Diff'] - (temp_df['loss'] * temp_df['CreditCard Diff'])
    actions['CreditCard Upper'] = temp_df['CreditCard Diff'] + (temp_df['loss'] * temp_df['CreditCard Diff'])
    temp_df = actions.reset_index().set_index('PaymentMethod_Bank transfer (automatic)').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['BankTransfer Lower'] = temp_df['BankTransfer Diff'] - (.5 * temp_df['loss'] * temp_df['BankTransfer Diff'])
    actions['BankTransfer Upper'] = temp_df['BankTransfer Diff'] + (.5 * temp_df['loss'] * temp_df['BankTransfer Diff'])
    temp_df = actions.reset_index().set_index('Contract_One year').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['1yrContract Lower'] = temp_df['1yrContract Diff'] - (.5 * temp_df['loss'] * temp_df['1yrContract Diff'])
    actions['1yrContract Upper'] = temp_df['1yrContract Diff'] + (.5 * temp_df['loss'] * temp_df['1yrContract Diff'])
    temp_df = actions.reset_index().set_index('Contract_Two year').join(loss_df)
    temp_df = temp_df.set_index('index')
    actions['2yrContract Lower'] = temp_df['2yrContract Diff'] - (.5 * temp_df['loss'] * temp_df['2yrContract Diff'])
    actions['2yrContract Upper'] = temp_df['2yrContract Diff'] + (.5 * temp_df['loss'] * temp_df['2yrContract Diff'])

    print("finished!")
