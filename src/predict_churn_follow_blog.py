"""
This script follows the procedure laid down in the blog:
https://towardsdatascience.com/hands-on-predict-customer-churn-5c2a42806266

It uses the Telecom churn dataset, which can be downloaded at:
https://www.kaggle.com/mnassrib/telecom-churn-datasets

Author: Hans Weda, Rond consulting
Date: 5 february 2021
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join("..", "data", "raw")
FIGURES_DIR = os.path.join("..", "reports", "figures")


def load_and_clean_data() -> pd.DataFrame:
    # Loading the CSV with pandas
    df = pd.read_csv(os.path.join(DATA_DIR, "churn-bigml-80.csv"))

    # create dummies
    df["Churn"] = df["Churn"].astype(int)

    #df = df.drop(labels=["State"], axis=1)
    df = pd.concat([df.drop(labels=["State"], axis=1), pd.get_dummies(df["State"], prefix="State")], axis="columns")

    for col in ["International plan", "Voice mail plan"]:
        df = pd.concat([df.drop(labels=[col], axis=1), pd.get_dummies(df[col], prefix=col, drop_first=True)], axis="columns")

    return df


def explorative_visualization(df: pd.DataFrame) -> None:
    # pairs plot
    # TODO: this visualization takes very long - refactor somehow
    sns_plot = sns.pairplot(df.sample(100, random_state=1))
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(FIGURES_DIR, "pairs_plot.png"))
    return


def plot_result(model, X_train, X_test, y_train, y_test, model_name: str = "Name of model") -> plt.Figure:
    prediction_test = model.predict(X_test)

    # Print the prediction accuracy
    print(metrics.accuracy_score(y_test, prediction_test))

    # To get the weights of all the variables
    if hasattr(model, "coef_"):
        weights = pd.Series(model.coef_[0], index=X_train.columns.values)
    elif hasattr(model, "feature_importances_"):
        weights = pd.Series(model.feature_importances_, index=X_train.columns.values)
    else:
        weights = pd.Series(0*y_train.values, index=X_train.columns.values)

    print(weights.sort_values(ascending=False))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    weights.abs().sort_values(ascending=False).head(10).reset_index(name="weight").plot.bar(
        x='index', y='weight',
        ax=axs[0],
        label="Absolute weight"
    )
    axs[0].set_xlabel("Feature")
    axs[0].set_ylabel("Weigth")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    acc = metrics.accuracy_score(y_test, prediction_test)
    axs[1].plot(fpr, tpr, label="churn-test, auc={:.3f}".format(auc))

    y_pred_proba = model.predict_proba(X_train)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_train,  y_pred_proba)
    auc = metrics.roc_auc_score(y_train, y_pred_proba)
    axs[1].plot(fpr, tpr, label="churn-train, auc={:.3f}".format(auc))
    axs[1].legend(loc=4)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title("Test accuracy: {:.3f}".format(acc))
    plt.suptitle(model_name)
    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    # loading and cleaning the data
    df = load_and_clean_data()

    # explorative visualization
    # explorative_visualization(df)

    # forecasting churn
    y = df["Churn"].values
    X = df.drop(labels=["Churn"], axis=1)

    # Create Train & Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # Logistic regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    fig = plot_result(model=lr_model,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                model_name="Logistic regression")
    fig.savefig(os.path.join(FIGURES_DIR, "logistic_regression.png"))

    # random forest
    # Instantiate model with 1000 decision trees
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf_model.fit(X_train, y_train)

    fig = plot_result(model=rf_model,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                model_name="Random forest")
    fig.savefig(os.path.join(FIGURES_DIR, "random_forest.png"))
