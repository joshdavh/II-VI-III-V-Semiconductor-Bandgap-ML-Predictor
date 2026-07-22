"""
This module contains functions for training and evaluating machine learning
models to predict the band gap of semiconductors based on their features.
Includes functions for model performance evaluation, plotting predicted
vs actual values, generating model graphs, performing SHAP analysis,
analyzing residuals, featurizing data, and predicting band gaps for specific
chemical formulas.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import shap
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

INPUT_PATH = "data/processed/mp_iii_v_ii_vi_feature.csv"

FEATURE_COLUMNS = [
    "MagpieData range Electronegativity",
    "MagpieData mean CovalentRadius",
]

# prints the mean and standard deviation of the three models' mean absolute error (MAE)
def print_model_performance(scores, scoresDummy, scoresXGB):
    print(f"Mean MAE: {scores.mean():.4f} eV")
    print(f"Std Dev of MAE: {scores.std():.4f} eV")
    
    print(f"Dummy Mean MAE: {scoresDummy.mean():.4f} eV")
    print(f"Dummy Regressor MAE: {scoresDummy.mean():.4f} eV")

    print(f"XGB Mean MAE: {scoresXGB.mean():.4f} eV")
    print(f"XGB Std Dev: {scoresXGB.std():.4f} eV")

# plots predicted vs actual band gap values for a given model and dataset
def run_predictvsactual(model, X, Y, loo):
    y_predict = cross_val_predict(model, X, Y, cv=loo)
    plt.scatter(Y, y_predict)
    plt.plot([0, Y.max()], [0, Y.max()], linestyle="--", color="gray")
    plt.xlabel("Actual Band Gap (eV)")
    plt.ylabel("Predicted Band Gap (eV)")
    plt.title("Predicted vs. Actual Band Gap")
    plt.savefig("figures/predicted_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.show()

# generates a contour plot of predicted band gap values over a grid of
# electronegativity range and mean covalent radius
def run_model_graph(model, X, Y):
    model.fit(X, Y)

    en_range_vals = np.linspace(X["MagpieData range Electronegativity"].min(), X["MagpieData range Electronegativity"].max(), 50)
    radius_vals = np.linspace(X["MagpieData mean CovalentRadius"].min(), X["MagpieData mean CovalentRadius"].max(), 50)
    EN_grid, RADIUS_grid = np.meshgrid(en_range_vals, radius_vals)

    X_grid = np.column_stack([EN_grid.ravel(), RADIUS_grid.ravel()])
    band_gap_pred_grid = model.predict(X_grid).reshape(EN_grid.shape)

    plt.contourf(EN_grid, RADIUS_grid, band_gap_pred_grid, levels=20, cmap="viridis")
    plt.colorbar(label="Predicted Band Gap (eV)")

    plt.scatter(
        X["MagpieData range Electronegativity"],
        X["MagpieData mean CovalentRadius"],
        c=Y,
        edgecolor="black",
        cmap="viridis",
    )

    plt.xlabel("Electronegativity Range")
    plt.ylabel("Mean Covalent Radius")
    plt.title("Predicted Band Gap Surface")
    plt.savefig("figures/modelgraph.png", dpi=300, bbox_inches="tight")
    plt.show()

# runs SHAP analysis on the model to determine feature importance and generates a summary plot
def run_shap_analysis(model, X, Y):
    model.fit(X, Y)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("figures/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
    return shap_values

# analyzes the residuals of the model predictions and prints the worst predictions
def analyze_residuals(df, y, y_pred, top_n=10):
    df_results = pd.DataFrame({
        "formula": df["formula"],
        "actual": y,
        "predicted": y_pred,
        "residual": y - y_pred,
    })
    df_results["abs_residual"] = df_results["residual"].abs()

    worst_predictions = df_results.sort_values("abs_residual", ascending=False)
    print(worst_predictions.head(top_n))

    return worst_predictions

# used for featurizing new formulas for manual prediction
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df["composition"] = df["formula"].apply(Composition)
    featurizer = ElementProperty.from_preset("magpie")
    df_featurized = featurizer.featurize_dataframe(df, col_id="composition")
    return df_featurized

# predicts the band gap for a given chemical formula using the trained model
def formula_predict(model, formula, X, Y):
    model.fit(X, Y)
    input_df = pd.DataFrame({"formula": [formula]})
    featurized_df = featurize(input_df)
    print(featurized_df[FEATURE_COLUMNS])
    X_new = featurized_df[FEATURE_COLUMNS]
    predicted_band_gap = model.predict(X_new)[0]
    print(f"Predicted band gap for {formula}: {predicted_band_gap:.4f} eV")
    return predicted_band_gap

def main():
    df = pd.read_csv(INPUT_PATH)

    X = df[FEATURE_COLUMNS]
    Y = df["band_gap"]

    
    model = LinearRegression()
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, Y, cv=loo, scoring="neg_mean_absolute_error")
    scores = -scores

    modelDummy = DummyRegressor()
    scoresDummy = cross_val_score(modelDummy, X, Y, cv=loo, scoring="neg_mean_absolute_error")
    scoresDummy = -scoresDummy
    
    modelXGB = XGBRegressor()
    scoresXGB = cross_val_score(modelXGB, X, Y, cv=loo, scoring="neg_mean_absolute_error")
    scoresXGB = -scoresXGB

    # PRINT GENERAL MODEL PERFORMANCE METRICS
    #print_model_performance(scores, scoresDummy, scoresXGB)

    # SELECT WHICH MODEL TO USE FOR PLOTTING
    model = modelXGB

    # RUN PREDICT VS ACTUAL GRAPH
    # run_predictvsactual(model, X, Y, loo)

    # RUN MODEL GRAPH
    # run_model_graph(model, X, Y)

    # RUN SHAP ANALYSIS
    # print(run_shap_analysis(model, X, Y))

    # ANALYZE RESIDUALS
    # y_predict = cross_val_predict(model, X, Y, cv=loo)
    # analyze_residuals(df, Y, y_predict, top_n=20)

    # INPUT FORMULAS FOR PREDICTION
    #formula_predict(model, input("Enter a chemical formula to predict its band gap: "), X, Y)

if __name__ == "__main__":
    main() 
