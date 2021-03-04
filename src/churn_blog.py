"""
script to generate the figures needed for the rond churn blog

Author: Hans Weda
Date: 12 february 2021
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.plotting import plot_lifetimes
from matplotlib.lines import Line2D
from numpy.random import default_rng
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from lifelines.calibration import survival_probability_calibration
from src.data.data_utils import load_from_kaggle
from src.models.model_utils import remove_collinear_variables
from src.survival_analysis_for_churn import clean_data
from sklearn.inspection import plot_partial_dependence
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import seaborn as sns


observation_idx = 100
idx_range = np.linspace(0, 12)

observation_date = pd.to_datetime("01-03-2021", format="%d-%m-%Y")

date_range = pd.date_range(
    start=observation_date - pd.DateOffset(years=1),
    end=observation_date,
    freq='MS',
    normalize=True
)

rng = default_rng(321)

FIGURES_DIR = os.path.join("..", "reports", "figures")


def _determine_end_idx(row):
    if row['Churn'] == "No":
        # not churned yet
        out = 100
    else:
        out = 100 - rng.integers(len(idx_range))
    return out


def plot_overview(df_plot: pd.DataFrame):
    """
    Generates plot showing survival times including forecasts

    Parameters:
        df_plot: a pandas dataframe consisting observed and predicted trajectories

    Returns:
        fig: a figure displaying survival times for a subset of customers
    """
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 5))
    plot_lifetimes(
        durations=df_plot["tenure"],
        event_observed=(df_plot["Churn"] == "Yes"),
        entry=df_plot["start_idx"],
        sort_by_duration=False,
        ax=ax_main
    )
    # create zombie lines for the legend
    zombie_lines = [
        Line2D([0, 1], [0, 1], color=[0.20392157, 0.54117647, 0.74117647, 1.]),
        Line2D([0, 1], [0, 1], color=[0.65098039, 0.02352941, 0.15686275, 1.])
    ]
    legend_labels = ['Existing customer', 'Churned customer']

    # add vertical dashed line
    ax_main.axvline(observation_idx, color='darkgrey', linestyle='--')
    if "remaining_life" in df_plot.columns:
        # plot remaining life
        ax_main.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 80)
        temp = df_plot[df_plot["Churn"] == "No"].copy()
        temp["remaining_life"] = temp.apply(
            lambda row: 180 if row["remaining_life"] > 80 else 100 + row["remaining_life"],
            axis=1
        )
        lc = ax_main.hlines(y=temp.index, xmin=100, xmax=temp["remaining_life"], linestyle='--', color="purple")
        idx_max = 172
        zombie_lines.append(lc)
        legend_labels.append('Forecasted membership')
    else:
        ax_main.set_xlim(ax_main.get_xlim()[0], ax_main.get_xlim()[1] + 20)
        ax_main.text(observation_idx + 10, sum(ax_main.get_ylim()) / 2, "?", fontsize=40)
        idx_max = 112

    # set legend
    ax_legend.legend(zombie_lines, legend_labels, loc="upper center")
    ax_legend.axis("off")
    # set axis labels
    ax_main.set_ylabel("Customer ID")
    ax_main.set_xlabel("Date")
    ax_main.set_xticks(np.arange(idx_max % 12, idx_max + 1, 12))
    ax_main.set_xticklabels(
        [date.strftime("%Y-%m-%d") for date in pd.date_range(
            end=observation_date + pd.DateOffset(years=(idx_max - 100) // 12),
            freq="12MS",
            periods=(idx_max // 12) + 1
        )],
        rotation=45,
        ha='right'
    )

    plt.tight_layout()
    return fig


def kaplan_meier_plots(df: pd.DataFrame) -> None:
    """
    Generates plot for Kaplan-Meier Curves.

    Parameters:
        df: a dataframe containing data
    Returns:
        None
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['tenure'], event_observed=df["Churn"] == "Yes")
    fig, ax = plt.subplots(1, 1)
    kmf.plot_survival_function(at_risk_counts=True, ax=ax)
    plt.tight_layout()
    ax.set_xlabel("Subscription time [months]")
    ax.set_ylabel("Subscription probability")
    fig.savefig(os.path.join(FIGURES_DIR, 'KaplanMeier_plot.png'))

    # survival analysis KaplanMeier per Contract duration
    fig, ax = plt.subplots(1, 1)
    for key, gr in df.groupby("Contract"):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=gr['tenure'], event_observed=gr["Churn"] == "Yes")
        kmf.plot_survival_function(ax=ax, label=key)
    ax.set_xlabel("Subscription time [months]")
    ax.set_ylabel("Subscription probability")
    ax.set_title('Kaplan-Meier Survival Curve by Contract Duration')
    fig.savefig(os.path.join(FIGURES_DIR, 'KaplanMeier_plot_vs_contract.png'))

    return


def brier_scores(df_test: pd.DataFrame, X_test, models: list, labels: list) -> plt.Figure:
    """
    Calculates Brier scores over the test set.

    Parameters:
        df_test: pandas DataFrame containing data on customers part of the test set
        X_test
        models: A list of survival models for which Brier scores need to be calculated
        labels:

    Returns:
        fig: a plot with Brier scores

    """


    loss_list = list()
    for i in range(1, 73):
        scores = list()
        for model in models:
            if "lifelines" in model.__module__:
                # lifelines model
                scores.append(
                    brier_score_loss(
                        y_true=df_test['Churn_Yes'],
                        y_prob=1 - model.predict_survival_function(df_test).loc[i].values,
                        pos_label=1
                    )
                )
            else:
                # sklearn survival model
                scores.append(
                    brier_score_loss(
                        y_true=df_test['Churn_Yes'],
                        y_prob=1 - model.predict_survival_function(X_test, return_array=True)[:, i-1],
                        pos_label=1
                    )
                )
        loss_list.append([i] + scores)
    loss_df = pd.DataFrame(loss_list, columns=["Time"] + labels).set_index("Time")
    fig, ax = plt.subplots()
    loss_df.plot(ax=ax)
    ax.set(xlabel='Prediction Time', ylabel='Calibration Loss', title='Calibration Loss by Time')
    return fig


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
    kaplan_meier_plots(df)

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

    fig, ax = plt.subplots(1, 1)
    cph.baseline_survival_.rename(
        columns={
            (0, 0): 'Monthly',
            (1, 0): 'Two year',
            (0, 1): 'One year'
        }).plot(ax=ax)
    ax.set_xlabel("Subscription time [months]")
    ax.set_ylabel("Subscription probability")
    ax.set_title('CoxPH Survival Curve by Contract Duration')
    fig.savefig(os.path.join(FIGURES_DIR, 'CoxPH_plot_vs_contract.png'))

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
