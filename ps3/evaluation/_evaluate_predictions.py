import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import mean_absolute_error, mean_squared_error, auc

def _gini_from_lorenz(actual, predicted, weight):
    """
    Compute the Gini coefficient using the same formula as in ps3_script:
    gini = 1 - 2 * area under the Lorenz curve
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    weight = np.asarray(weight)

    # Order by predicted pure premium (ascending risk)
    ranking = np.argsort(predicted)

    ranked_actual = actual[ranking]
    ranked_weight = weight[ranking]

    # Cumulative claim amount weighted by exposure
    cum_claim = np.cumsum(ranked_actual * ranked_weight)
    cum_claim = cum_claim / cum_claim[-1]  # normalize to 1

    # Cumulative share of policyholders (0 → 1)
    cum_samples = np.linspace(0, 1, len(cum_claim))

    # Area under Lorenz curve
    lorenz_area = auc(cum_samples, cum_claim)

    # Gini definition from ps3_script
    gini = 1 - 2 * lorenz_area
    return gini


def evaluate_predictions(actual, predicted, weight):
    """
    Evaluate model predictions against actual outcomes using exposure weights.

    Returns bias, deviance, MAE, RMSE, and Gini.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    weight = np.asarray(weight)

    # ------------------------------------
    # 1. Bias
    # ------------------------------------
    actual_mean = np.average(actual, weights=weight)
    pred_mean   = np.average(predicted, weights=weight)
    bias = pred_mean - actual_mean

    # ------------------------------------
    # 2. Deviance (Tweedie)
    # ------------------------------------
    TweedieDist = TweedieDistribution(1.5)
    deviance = TweedieDist.deviance(
        actual,
        predicted,
        sample_weight=weight,
    ) / np.sum(weight)

    # ------------------------------------
    # 3. MAE and RMSE
    # ------------------------------------
    mae = mean_absolute_error(actual, predicted, sample_weight=weight)
    mse = mean_squared_error(actual, predicted, sample_weight=weight)
    rmse = np.sqrt(mse)
    
    # ------------------------------------
    # 4. Gini (Bonus)
    # ------------------------------------
    gini = _gini_from_lorenz(actual, predicted, weight)

    # ------------------------------------
    # 5. Return DataFrame with metric names as index
    # ------------------------------------
    metrics = {
        "bias": bias,
        "deviance": deviance,
        "MAE": mae,
        "RMSE": rmse,
        "Gini": gini,
    }

    return pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
