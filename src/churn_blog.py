"""
script to generate the figures needed for the rond churn blog

Author: Hans Weda
Date: 12 february 2021
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lifelines.plotting import plot_lifetimes
from numpy.random import default_rng

from src.data.data_utils import load_from_kaggle
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

rng = default_rng()


def _determine_end_idx(row):
    if row['Churn_Yes'] == 0:
        # not churned yet
        out = 100
    else:
        out = 100 - rng.integers(len(idx_range))
    return out


if __name__ == "__main__":
    # loading from kaggle
    df = load_from_kaggle(owner="blastchar", dataset_name="telco-customer-churn")
    # clean the data
    df = clean_data(df)
    df["end_idx"] = df.apply(_determine_end_idx, axis=1)
    df["start_idx"] = df["end_idx"] - df["tenure"]
    print(df.head())
    fig, ax = plt.subplots(1, 1)
    df_plot = df.head(20)
    plot_lifetimes(
        durations=df_plot["tenure"],
        event_observed=df_plot["Churn_Yes"],
        entry=df_plot["start_idx"],
        sort_by_duration=False,
        ax=ax
    )
    ax.set_ylabel("Customer ID")
    ax.set_xlabel("Date")
    ax.set_xticks(np.arange(observation_idx % 12, observation_idx+1, 12))
    ax.set_xticklabels(
        [date.strftime("%Y-%m-%d") for date in pd.date_range(
            end=observation_date,
            freq="12MS",
            periods=(observation_idx//12)+1
        )],
        rotation=45,
        ha='right'
    )
    ax.legend([ax.get_children()[0]], ['test'])
    """
    def todate(x, pos, today=pd.Timestamp.today()):
        return (today + pd.DateOffset(months=x-100)).strftime("%Y-%m-%d")

    fmt = ticker.FuncFormatter(todate)
    ax.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate(rotation=45)
    """
    plt.tight_layout()
    plt.show()
    print("finished!")
