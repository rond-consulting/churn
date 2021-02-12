import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings


def remove_collinear_variables(data: pd.DataFrame, target: str, thresh: float = 5.0) -> pd.DataFrame:
    """
    Removes highly collinear variables from the dataframe
    """
    variables = data.drop(columns=[target]).columns.to_list()
    dropped = True
    while dropped:
        dropped = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", lineno=193)
            vifs = get_VIF(data[variables])

        if vifs.max() > thresh:
            print('VIF: {}, dropping column: {}'.format(vifs.max(), vifs.idxmax()))
            variables.remove(vifs.idxmax())
            dropped = True

    print('Remaining variables:')
    print(variables)
    return data[variables + [target]]


def get_VIF(data: pd.DataFrame) -> float:
    out = pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])], index=data.columns)
    return out
