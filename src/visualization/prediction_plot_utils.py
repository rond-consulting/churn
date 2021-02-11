import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


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
    axs[0].set_title("Top 10 important features")

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
