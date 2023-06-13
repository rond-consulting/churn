import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

# load the rossi data as example of a churn data-set
rossi = load_rossi()

rossi.to_csv(os.path.join("..", "data", "raw", "rossi.csv"), index_label="index")

# create simple model without optimalisation
cph = CoxPHFitter()
cph.fit(rossi, duration_col='week', event_col='arrest', formula="fin + wexp + age * prio")

cph.print_summary()  # access the individual results using cph.summary

cph.plot()

# save model to disk
with open(os.path.join("..", "deployments", "rossi_survival", "survival_model.pkl"), "wb") as f:
    pickle.dump(cph, f)

# show results
preds = cph.predict_percentile(rossi, p=0.8)
print(preds)
with pd.option_context('mode.use_inf_as_na', True):
    print(preds.fillna(-999))
plt.show()
print("finished!")
