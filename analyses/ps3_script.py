# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

from ps3.evaluation._evaluate_predictions import evaluate_predictions


# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?

# Answer: ClaimAmountCut is a total amount over the exposure period. Since policies have different
# levels of exposure, dividing by exposure can standardize the
# outcome. We can use PurePremium to model the claim rate.

# TODO: use your create_sample_split function here
df = create_sample_split(df, id_column="IDpol", training_frac=0.8)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]

numeric_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("spline", SplineTransformer(knots="quantile", include_bias=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("estimate", GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.
model_pipeline = Pipeline(
    steps=[
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=0,
            ),
        )
    ]
)

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
lgbm_pipeline = Pipeline(
    steps=[
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=0,
            ),
        )
    ]
)

param_grid = {
    "estimate__learning_rate": [0.01, 0.05, 0.1],
    "estimate__n_estimators": [200, 500, 800],
}

cv = GridSearchCV(
    estimator=lgbm_pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,

)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()



# %%
# ==============================================================================
# START OF PROBLEM SET 4
# ==============================================================================

# ------------------------------------------------------------------------------
# PS4 Ex 1: Monotonicity Constraints
# ------------------------------------------------------------------------------

# 1. Plot average claims per BonusMalus group
# -------------------------------------------------------
# Group by BonusMalus and calculate weighted average PurePremium
df_plot = (
    df.groupby("BonusMalus")
    .apply(lambda x: pd.Series({
        "WeightedPurePremium": x["ClaimAmountCut"].sum() / x["Exposure"].sum(),
        "Exposure": x["Exposure"].sum()
    }))
    .reset_index()
)

# For clarity, filter out data points with low exposure (Exposure < 100) 
# to avoid extreme noise in the plot.
df_plot_filtered = df_plot[df_plot["Exposure"] > 100]

plt.figure(figsize=(10, 6))
plt.scatter(df_plot_filtered["BonusMalus"], df_plot_filtered["WeightedPurePremium"], 
            alpha=0.6, label="Observed Average")
plt.plot(df_plot_filtered["BonusMalus"], df_plot_filtered["WeightedPurePremium"], 
         alpha=0.3, color='blue') # Connect lines to visualize the trend

plt.xlabel("BonusMalus Score")
plt.ylabel("Weighted Pure Premium")
plt.title("Observed Pure Premium per BonusMalus Level")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Quesiton1: What will/could happen if we do not include a monotonicity constraint?
# Answer: Without constraints, tree models might capture local data fluctuations (noise).
# For example, if people with BonusMalus=105 happen to be lucky and have no claims 
# in the training set, the model might think 105 is lower risk than 100. 
# This violates the logical expectation of insurance pricing.

# %%
# 2. Define Monotonicity Constraints
# -------------------------------------------------------
# The monotone_constraints parameter in LightGBM requires a list.
# List length = number of features.
# 0 = no constraint, 1 = increasing, -1 = decreasing.

# Get feature name list (based on previous X_train_t)
feature_names = X_train_t.columns.tolist()

# Initialize a list of zeros (no constraints)
monotone_constraints = [0] * len(feature_names)

# Find the index of BonusMalus
bm_index = feature_names.index("BonusMalus")

# Set BonusMalus to 1 (increasing monotonicity constraint)
monotone_constraints[bm_index] = 1

print(f"Feature list: {feature_names}")
print(f"Constraint list: {monotone_constraints}")

# %%
# 3. Train Constrained LGBM with Cross-Validation
# -------------------------------------------------------
# Create a new Pipeline named constrained_lgbm

# Define LGBM Regressor with constraints (inner estimator)
lgbm_estimator = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    monotone_constraints=monotone_constraints, # <--- Key change: Add constraints
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    verbose=-1
)

# Create the pipeline called 'constrained_lgbm' 
constrained_lgbm = Pipeline([
    ("estimate", lgbm_estimator)
])

# Use the same parameter grid as before for fair comparison
param_grid = {
    "estimate__learning_rate": [0.01, 0.05, 0.1],
    "estimate__n_estimators": [200, 500, 800],
}

cv_constrained = GridSearchCV(
    estimator=constrained_lgbm, 
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)

# Train the model
print("Training constrained LGBM...")
cv_constrained.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

# %%
# 4. Predict and Evaluate
# -------------------------------------------------------
# Save predictions to the pp_t_lgbm_constrained column
df_test["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_train_t)

# Calculate and print Deviance (Testing Loss)
constrained_loss = TweedieDist.deviance(
    y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t
) / np.sum(w_test_t)

print(f"Best params (Constrained): {cv_constrained.best_params_}")
print(f"Testing loss (Constrained LGBM): {constrained_loss}")

# Comparison:
# Usually, the Constrained Loss might be slightly higher or similar to the 
# Unconstrained Loss, but the model now adheres to stricter business logic.


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



# %%
# ------------------------------------------------------------------------------
# PS4 Ex 2: Learning Curve
# ------------------------------------------------------------------------------

import lightgbm as lgb # Needed for the plotting function

# 1. Re-fit the best estimator with eval_set
# -------------------------------------------------------
# We need to extract the actual LGBMRegressor object from the Pipeline
# because the Pipeline's .fit() method does not accept 'eval_set'.

# Extract the 'estimate' step from the best pipeline found in Ex 1
# This object already has the best hyperparameters (learning_rate=0.01, n_estimators=500)
best_lgbm_model = cv_constrained.best_estimator_.named_steps["estimate"]

print("Re-fitting the best model with eval_set to track convergence...")

# We fit the model again. This time we provide:
# - eval_set: A list of (X, y) pairs to evaluate during training.
# - eval_names: Labels for the plot.
# - eval_sample_weight: Crucial for insurance! We must weigh validation by exposure.
# - eval_metric: 'tweedie' to track the Deviance.
best_lgbm_model.fit(
    X_train_t, y_train_t,
    sample_weight=w_train_t,
    eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    eval_names=['Train', 'Test'],
    eval_metric='tweedie',
    eval_sample_weight=[w_train_t, w_test_t]
)

# 2. Plot the learning curve
# -------------------------------------------------------
# Use LightGBM's built-in plotting function to visualize the metric (tweedie deviance)
# over the number of boosting iterations (trees).

print("Plotting learning curve...")
ax = lgb.plot_metric(best_lgbm_model, metric='tweedie', figsize=(10, 6))
plt.title("Learning Curve (Tweedie Deviance)")
plt.ylabel("Tweedie Deviance")
plt.xlabel("Number of Estimators (Trees)")
plt.grid(True, alpha=0.3)
plt.show()

# 3. Question 3
# -------------------------------------------------------
# What do you notice, is the estimator tuned optimally?
#
# The Training loss (blue line) decreases continuously, indicating the model keeps learning from the training data.
# The Test loss (orange line) decreases initially but reaches a minimum around 280–300 trees. 
# After that point, it plateaus and even slightly increases towards 500.
# The estimator is not stricly optimal, as the model is slightly overfitting at 500 estimators. 
# Since the test error stops improving after ~300 trees while the training error keeps dropping, 
# the additional 200 trees are fitting noise rather than signal.


# %%
# ------------------------------------------------------------------------------
# PS4 Ex 3(Q8): Compare constrained vs unconstrained LGBM using evaluation metrics
# ------------------------------------------------------------------------------

# Unconstrained LGBM metrics (from tuned cv.best_estimator_)
metrics_unconstrained = evaluate_predictions(
    actual=y_test_t,
    predicted=df_test["pp_t_lgbm"],
    weight=w_test_t,
)
metrics_unconstrained.columns = ["unconstrained_lgbm"]

# Constrained LGBM metrics (from cv_constrained.best_estimator_)
metrics_constrained = evaluate_predictions(
    actual=y_test_t,
    predicted=df_test["pp_t_lgbm_constrained"],
    weight=w_test_t,
)
metrics_constrained.columns = ["constrained_lgbm"]

# Combine into a single comparison table
metrics_comparison = metrics_unconstrained.join(metrics_constrained)
metrics_comparison.index.name = "Metrics"

print("\n" + "="*60)
print("Model Performance Comparison")
print("="*60)
print(metrics_comparison)
print("="*60 + "\n")


# -------------------------------------------------------
#Interpretation:

#The unconstrained LGBM performs slightly better across all predictive metrics:
#it has lower deviance, MAE, and RMSE, and a higher Gini coefficient. This means
# the unconstrained model captures the nonlinear patterns in the data more flexibly 
# and ranks policyholders more accurately by risk.

#The constrained model performs very close to the unconstrained one, but its
#predictions respect the required monotonicity with respect to BonusMalus. This
#consistency is often more important in pricing than the small loss in accuracy.

#Overall, the unconstrained model is marginally more accurate, but the constrained
#model is more appropriate for insurance pricing because it enforces monotonicity
#while maintaining almost identical predictive performance.

# %%
# ------------------------------------------------------------------------------
# PS4 Ex 4: Evaluation plots (Partial Dependence Plots)
# ------------------------------------------------------------------------------
import dalex as dx

# Step 1: Define explainer objects for both unconstrained and constrained LGBM models
# -------------------------------------------------------
# For the unconstrained LGBM
exp_unconstrained = dx.Explainer(
    cv.best_estimator_,
    data=X_test_t,
    y=y_test_t,
    label="Unconstrained LGBM",
    verbose=False
)

# For the constrained LGBM
exp_constrained = dx.Explainer(
    cv_constrained.best_estimator_,
    data=X_test_t,
    y=y_test_t,
    label="Constrained LGBM",
    verbose=False
)

print("Explainers created successfully!")

# %%
# Step 2: Compute marginal effects using model_profile and plot PDPs for all features
# -------------------------------------------------------
# Get all feature names
all_features = X_test_t.columns.tolist()
print(f"Computing PDPs for {len(all_features)} features...")

# Compute PDPs for unconstrained model
pdp_unconstrained = exp_unconstrained.model_profile(
    variables=all_features,
    verbose=False
)

# Compute PDPs for constrained model
pdp_constrained = exp_constrained.model_profile(
    variables=all_features,
    verbose=False
)

# Plot and compare PDPs between the two models
print("Plotting Partial Dependence Plots...")
fig = pdp_unconstrained.plot(pdp_constrained, show=False)

fig.update_layout(
    title="Partial Dependence Plots: Unconstrained vs Constrained LGBM",
    width=1200,
    height=800,
    margin=dict(l=100, r=100, t=100, b=100),
)

fig.show()

# %%
# ------------------------------------------------------------------------------
# PS4 Ex 5: Shapley
# ------------------------------------------------------------------------------

# 1. Select a specific observation
observation = X_test_t.iloc[[0]]
print("Observation features:")
print(observation.T)

# 2. Define DALEX Explainers
exp_glm = dx.Explainer(
    t_glm1,
    data=X_test_t,
    y=y_test_t,
    label="GLM Benchmark",
    verbose=False
)

exp_lgbm = dx.Explainer(
    cv_constrained.best_estimator_,
    data=X_test_t,
    y=y_test_t,
    label="Constrained LGBM",
    verbose=False
)

# 3. Calculate SHAP values
shap_glm = exp_glm.predict_parts(observation, type="shap", B=25)
shap_lgbm = exp_lgbm.predict_parts(observation, type="shap", B=25)

# 4. Plot
fig = shap_glm.plot(shap_lgbm, max_vars=10, show=False)

fig.update_layout(
    width=900,  
    height=600, 
    
    margin=dict(l=200, r=100, t=80, b=50),
)
fig.show()

# Comment: BonusMalus is the single most important feature for both models, significantly pulling the predicted premium down.
# GLM is more sensitive compare to LGBM in many cases. For example, the bonus-malus score of GLM is -69.486, while the LGBM is -51.086.
# GLM relies heavily on Area = E (+24.104) as a major risk factor, while the LGBM places higher importance on Density (+10.08)
# %%
