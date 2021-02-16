"""
script to generate the figures needed for the rond churn blog

Author: Hans Weda
Date: 12 february 2021
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import plot_lifetimes
from matplotlib.lines import Line2D
from numpy.random import default_rng
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from lifelines.calibration import survival_probability_calibration
from src.data.data_utils import load_from_kaggle
from src.models.model_utils import remove_collinear_variables
from src.survival_analysis_for_churn import clean_data

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


def plot_overview(df_plot):
    fig, ax = plt.subplots(1, 1)
    plot_lifetimes(
        durations=df_plot["tenure"],
        event_observed=(df_plot["Churn"] == "Yes"),
        entry=df_plot["start_idx"],
        sort_by_duration=False,
        ax=ax
    )
    # create zombie lines for the legend
    lin1 = Line2D([0, 1], [0, 1], color=[0.20392157, 0.54117647, 0.74117647, 1.])
    lin2 = Line2D([0, 1], [0, 1], color=[0.65098039, 0.02352941, 0.15686275, 1.])
    ax.legend([lin1, lin2], ['Existing customer', 'Churned customer'], loc="lower left")

    # add vertical dashed line
    ax.axvline(observation_idx, color='darkgrey', linestyle='--')
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]+80)
    if "remaining_life" in df_plot.columns:
        # plot remaining life
        temp = df_plot[df_plot["Churn"] == "No"].copy()
        temp["remaining_life"] = temp.apply(lambda row: 180 if row["remaining_life"] > 80 else 100+row["remaining_life"], axis=1)
        ax.hlines(y=temp.index, xmin=100, xmax=temp["remaining_life"], linestyle='--', color="purple")
    else:
        ax.text(observation_idx+10, sum(ax.get_ylim())/2, "?", fontsize=40)

    # set axis labels
    ax.set_ylabel("Customer ID")
    ax.set_xlabel("Date")
    idx_max = 112
    ax.set_xticks(np.arange(idx_max % 12, idx_max+1, 12))
    ax.set_xticklabels(
        [date.strftime("%Y-%m-%d") for date in pd.date_range(
            end=observation_date + pd.DateOffset(years=1),
            freq="12MS",
            periods=(idx_max//12)+1
        )],
        rotation=45,
        ha='right'
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # loading from kaggle
    df = load_from_kaggle(owner="blastchar", dataset_name="telco-customer-churn")
    #df = df[df["Contract"] == "One year"]
    df["end_idx"] = df.apply(_determine_end_idx, axis=1)
    df["start_idx"] = df["end_idx"] - df["tenure"]
    print(df.head())
    fig = plot_overview(df.head(20))
    fig.savefig(os.path.join(FIGURES_DIR, "overview_data.png"))

    # survival analysis KaplanMeier
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

    # clean the data
    df_surv = clean_data(df)

    # prepare data for fitting
    df_surv = remove_collinear_variables(
        df_surv,
        target="Churn_Yes",
        thresh=100
    )

    cph = CoxPHFitter()
    cph_train, cph_test = train_test_split(df_surv, test_size=0.2)
    cph.fit(cph_train, 'tenure', 'Churn_Yes', strata=["Contract_Two year", "Contract_One year"])
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

    cph.check_assumptions(cph_train)

    # brier loss curve
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
    fig.savefig(os.path.join(FIGURES_DIR, 'Brier_score.png'))

    df_temp = cph_test.copy()
    df_temp["tenure"] = df_temp["tenure"] + 0.01
    fig, ax = plt.subplots()
    survival_probability_calibration(model=cph, df=df_temp, t0=25, ax=ax)
    fig.savefig(os.path.join(FIGURES_DIR, 'Calibration_curve.png'))

    # predict remaining time
    last_obs = df_surv.apply(lambda row: row['tenure'] if row["Churn_Yes"]==0 else 0, axis=1)

    # predict median remaining life
    remaining_life = cph.predict_median(df_surv, conditional_after=last_obs)
    remaining_life.name = "remaining_life"

    fig = plot_overview(df_plot=df.join(remaining_life).head(20))
    fig.savefig(os.path.join(FIGURES_DIR, "overview_prediction.png"))

    plt.show()

    # result
    print(cph.print_summary())
    cph.plot()
    plt.tight_layout()
    plt.show()
    print("finished!")

    # https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html?highlight=best#prediction-on-censored-subjects

