"""
Converts compositions into ML-ready numerical features using matminer's
magpie composition-based featurizer, focusing on features of mean radius
and electronegativity range. Saves the featurized data to a CSV file for
later use in model training.
"""

import os
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

INPUT_PATH = "data/raw/mp_iii_v_ii_vi_all.csv"
OUTPUT_PATH = "data/processed/mp_iii_v_ii_vi_feature.csv" 

SELECTED_FEATURES = [
    "MagpieData range Electronegativity",
    "MagpieData mean CovalentRadius",
]

# add more features as needed
# utilizes magpie to find elemental properties and featurize the compositions
# to machine learning-ready numerical features. focuses on mean radius and
# electronegativity range.
def featurize(df: pd.DataFrame):
    df["composition"] = df["formula"].apply(Composition)
    featurizer = ElementProperty.from_preset("magpie")
    df_featurized = featurizer.featurize_dataframe(df, col_id="composition")
    return df_featurized

def main():
    df = pd.read_csv(INPUT_PATH)
    df_featurized = featurize(df)

    df_final = df_featurized[["formula", "band_gap"] + SELECTED_FEATURES]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df_final)} compounds with {len(SELECTED_FEATURES)} features to {OUTPUT_PATH}")
    print(df_final.head())

if __name__ == "__main__":
    main()