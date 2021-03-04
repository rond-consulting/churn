"""
script to generate the figures needed for the rond churn blog

Author: Hans Weda
Date: 12 february 2021
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter

from numpy.random import default_rng

from sklearn.model_selection import train_test_split
from lifelines.calibration import survival_probability_calibration
from src.data.data_utils import load_from_kaggle
from src.visualization.generate_blog_plots import plot_overview, brier_scores, \
    kaplan_meier_plots, plot_coxph_stratified_survival_functions
from src.survival_analysis_for_churn import clean_data
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import seaborn as sns



idx_range = np.linspace(0, 12)
rng = default_rng(321)
FIGURES_DIR = os.path.join("..", "reports", "figures")


def _determine_end_idx(row):
    if row['Churn'] == "No":
        # not churned yet
        out = 100
    else:
        out = 100 - rng.integers(len(idx_range))
    return out

if __name__ == "__main__":
    # loading from kaggle
    df = load_from_kaggle(owner="blastchar", dataset_name="telco-customer-churn")
    # overview
    df["end_idx"] = df.apply(_determine_end_idx, axis=1)
    df["start_idx"] = df["end_idx"] - df["tenure"]
    print(df.head())
    fig = plot_overview(df.head(20))
    fig.savefig(os.path.join(FIGURES_DIR, "overview_data.png"))

    # Kaplan-Meier
    kaplan_meier_plots(df, FIGURES_DIR)

    # clean the data
    df_surv = clean_data(df)

    '''
    # prepare data for fitting
    df_surv = remove_collinear_variables(
        df_surv,
        target="Churn_Yes",
        thresh=100
    )
    '''

    # splitting
    random_state=468
    ## Maarten: Wat is het verschil tussen deze twee train_test_splits?
    df_train, df_test = train_test_split(df_surv, test_size=0.2, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        df_surv.drop(columns=["tenure", "Churn_Yes"]),
        Surv.from_dataframe(time="tenure", event="Churn_Yes", data=df_surv),
        test_size=0.2,
        random_state=random_state
    )

    # Cox PH fitting
    cph = CoxPHFitter()
    cph.fit(df_train, 'tenure', 'Churn_Yes', strata=["Contract_Two year", "Contract_One year"])
    print("Concordance index Cox PH: {}".format(cph.score(df_test, scoring_method="concordance_index")))
    plot_coxph_stratified_survival_functions(cph, FIGURES_DIR)

    cph.check_assumptions(df_train)

    # Weibull fitting
    waft = WeibullAFTFitter()
    waft.fit(df_train, 'tenure', 'Churn_Yes')
    print("Concordance index Weibull AFT: {}".format(waft.score(df_test, scoring_method="concordance_index")))

    fig, ax = plt.subplots(1, 1)
    waft.plot_partial_effects_on_outcome(
        ["Contract_One year", "Contract_Two year"],
        values=[[0, 0], [1, 0], [0, 1]],
        plot_baseline=False,
        ax=ax
    )
    ax.set_xlabel("Subscription time [months]")
    ax.set_ylabel("Subscription probability")
    ax.set_title('Weibull AFT Survival Curve by Contract Duration')
    ax.legend(labels=["Monthly", "One year", "Two year"])
    fig.savefig(os.path.join(FIGURES_DIR, 'WeibullAFT_plot_vs_contract.png'))

    # random forest survival
    rsf = RandomSurvivalForest(n_estimators=10,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)
    print("Concordance index random forest: {}".format(rsf.score(X_test, y_test)))

    fig, ax = plt.subplots(1, 1)
    for one, two in [(0, 0), (1, 0), (0, 1)]:
        X_temp = X_train
        X_temp["Contract_One year"] = one
        X_temp["Contract_Two year"] = two
        surv = rsf.predict_survival_function(X_temp, return_array=True)
        ax.plot(surv.mean(axis=0))
    ax.set_xlabel("Subscription time [months]")
    ax.set_ylabel("Subscription probability")
    ax.set_title('Random Survival Forest Survival Curve by Contract Duration')
    ax.legend(labels=["Monthly", "One year", "Two year"])
    fig.savefig(os.path.join(FIGURES_DIR, 'SurvivalForest_plot_vs_contract.png'))

    # brier scores
    fig = brier_scores(df_test, X_test, [cph, waft, rsf], ["Cox PH", "Weibull AFT", "Random forest"])
    fig.savefig(os.path.join(FIGURES_DIR, 'Brier_score.png'))

    fig, ax = plt.subplots()
    survival_probability_calibration(model=cph, df=df_test, t0=25, ax=ax)
    fig.savefig(os.path.join(FIGURES_DIR, 'Calibration_curve_cph.png'))

    fig, ax = plt.subplots()
    survival_probability_calibration(model=waft, df=df_test, t0=25, ax=ax)
    fig.savefig(os.path.join(FIGURES_DIR, 'Calibration_curve_waft.png'))

    # predict remaining time
    last_obs = df_surv.apply(lambda row: row['tenure'] if row["Churn_Yes"] == 0 else 0, axis=1)

    # predict median remaining life
    remaining_life = waft.predict_median(df_surv, conditional_after=last_obs)
    remaining_life.name = "remaining_life"

    fig = plot_overview(df_plot=df.join(remaining_life).head(20))
    
    fig.savefig(os.path.join(FIGURES_DIR, "overview_prediction.png"))

    # result
    print(waft.print_summary())
    fig, ax = plt.subplots()
    waft.plot()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'coefficients_waft.png'))
    plt.show()

    upgrades = [
        {"Contract_Two year": 1, "Contract_One year": 0},
        {"InternetService_No": 1},
        {"OnlineSecurity_Yes": 1},
        {"PaymentMethod_Electronic check": 0,
         "PaymentMethod_Mailed check": 0,
         "PaymentMethod_Credit card (automatic)": 0},
        {"OnlineBackup_Yes": 1}
    ]
    upgrade_labels = [
        "Contract duration of two years",
        "InternetService downgrade",
        "Adding online security",
        "Payment by automatic bank transfer",
        "Adding online backup"
    ]

    results = list()
    for upgrade, upgrade_label in zip(upgrades, upgrade_labels):
        # customer remaining life default
        remaining_life = waft.predict_median(df_surv, conditional_after=last_obs)
        remaining_life.name = "remaining_life"
        df_upgrade = df.join(remaining_life)

        # customer remaining life after upgrade
        df_surv_upgrade = df_surv.copy()
        for key in upgrade:
            df_surv_upgrade[key] = upgrade[key]
        remaining_life_upgrade = waft.predict_median(df_surv_upgrade, conditional_after=last_obs)
        remaining_life_upgrade.name = "remaining_life_upgrade"

        df_upgrade = df_upgrade.join(remaining_life_upgrade)

        fig, ax = plt.subplots()
        sns.kdeplot(df_upgrade[df_upgrade["Churn"] == "No"]["remaining_life"], fill=True, ax=ax, label="Unchanged")
        sns.kdeplot(df_upgrade[df_upgrade["Churn"] == "No"]["remaining_life_upgrade"], fill=True, ax=ax, label="Upgrade")
        ax.legend()
        ax.set_xlabel("Remaining subscription [months]")
        fig.savefig(os.path.join(FIGURES_DIR, 'life_{}.png'.format(upgrade_label)))

        # customer value
        df_upgrade["CustomerValue"] = df_upgrade["remaining_life"] * df_upgrade["MonthlyCharges"]
        df_upgrade["CustomerValueUpgrade"] = df_upgrade["remaining_life_upgrade"] * df_upgrade["MonthlyCharges"]

        results.append({
            "label": "Unchanged",
            "remaining_life": df_upgrade[df_upgrade["remaining_life"] < 12]["remaining_life"].mean(),
            "CustomerValue": df_upgrade[df_upgrade["remaining_life"] < 12]["CustomerValue"].mean(),
        })
        results.append({
            "label": upgrade_label,
            "remaining_life": df_upgrade[df_upgrade["remaining_life"] < 12]["remaining_life_upgrade"].mean(),
            "CustomerValue": df_upgrade[df_upgrade["remaining_life"] < 12]["CustomerValueUpgrade"].mean()
        })

        print(
            df_upgrade[df_upgrade["remaining_life"] < 12][["remaining_life", "remaining_life_upgrade"]].describe()
        )

        print(
            df_upgrade[df_upgrade["remaining_life"] < 12][["CustomerValue", "CustomerValueUpgrade"]].describe()
        )

    fig, ax = plt.subplots()
    pd.DataFrame(results).drop_duplicates().set_index('label').plot.barh(y='CustomerValue', ax=ax, legend=None)
    ax.set_xlabel("Customer Value")
    ax.set_ylabel("")
    ax.set_title("Customer value\nby different contract upgrades")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'upgrade_effects.png'))

    print("finished!")

    # https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html?highlight=best#prediction-on-censored-subjects
