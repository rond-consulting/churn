import matplotlib.pyplot as plt
import pandas as pd
from lifelines.plotting import plot_lifetimes
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import brier_score_loss


observation_idx = 100
observation_date = pd.to_datetime("01-03-2021", format="%d-%m-%Y")
date_range = pd.date_range(
    start=observation_date - pd.DateOffset(years=1),
    end=observation_date,
    freq='MS',
    normalize=True
)
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
        ax_main.set_xlim(ax_main.get_xlim()[0], ax_main.get_xlim()[1] + 80)
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

def brier_scores(df_test: pd.DataFrame, X_test, models: list, labels: list) -> plt.Figure:
    """
    Calculates Brier score losses over the test set.
    In a survival setting, Brier scores represent the average squared distance between
    the observed survival status and the predicted survival probability.

    Parameters:
        df_test: pandas DataFrame containing test data on customers with ALL columns
        X_test:  pandas DataFrame containing test data on customers only with feature columns (no survival time or event indicator)
        models: A list of survival models for which Brier scores need to be calculated
        labels: Labels for the legends
    Returns:
        fig: a plot with Brier scores losses for each inputted model
    """

    loss_list = list()
    max_tenure = 73

    for i in range(1, max_tenure):
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