"""
Pulls III-V and II-VI semiconductor compounds from Materials Project
using the current mp-api client

  - Binary III-V and II-VI compounds (e.g. GaAs, ZnSe)
  - Ternary isovalent alloys on a single sublattice:
        III-III-V  (e.g. AlGaAs)
        III-V-V    (e.g. AlNP)
        II-II-VI   (e.g. ZnCdS)
        II-VI-VI   (e.g. ZnSTe)
"""

import os
from itertools import combinations

import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# element sets defining the material family
GROUP_III = ["Al", "Ga", "In"]
GROUP_V = ["N", "P", "As", "Sb"]
GROUP_II = ["Zn", "Cd"]
GROUP_VI = ["S", "Se", "Te"]

# Binary pairs
ALLOWED_PAIRS = (
    [(a, b) for a in GROUP_III for b in GROUP_V]
    + [(a, b) for a in GROUP_II for b in GROUP_VI]
)

# Ternary combinations
III_III_V = [
    (a, b, c)
    for (a, b) in combinations(GROUP_III, 2)
    for c in GROUP_V
]
III_V_V = [
    (a, b, c)
    for a in GROUP_III
    for (b, c) in combinations(GROUP_V, 2)
]
II_II_VI = [
    (a, b, c)
    for (a, b) in combinations(GROUP_II, 2)
    for c in GROUP_VI
]
II_VI_VI = [
    (a, b, c)
    for a in GROUP_II
    for (b, c) in combinations(GROUP_VI, 2)
]
ALL_TERNARY_COMBOS = III_III_V + III_V_V + II_II_VI + II_VI_VI

BANDGAP_MIN = 0.01   # exclude exact zero (conductive materials)
BANDGAP_MAX = 6.0    # upper bound of band gap for semiconductors of interest

OUTPUT_PATH = "data/raw/mp_iii_v_ii_vi_all.csv"

FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
    "is_gap_direct",
    "symmetry",
    "energy_above_hull",
    "formation_energy_per_atom",
    "density",
    "volume",
    "nsites",
]


def _query_chemsys(mpr, chemsys, elements_tag):
    """
    Queries the Materials Project API for chemical systems (chemsys) and returns a
    list of dictionaries containing relevant with composition
    """
    docs = mpr.materials.summary.search(
        chemsys=chemsys,
        band_gap=(BANDGAP_MIN, BANDGAP_MAX),
        fields=FIELDS,
    )

    rows = []
    for d in docs:
        rows.append(
            {
                "material_id": d.material_id,
                "formula": d.formula_pretty,
                "elements": elements_tag,
                "band_gap": d.band_gap,
                "is_gap_direct": d.is_gap_direct,
                "spacegroup_symbol": d.symmetry.symbol if d.symmetry else None,
                "crystal_system": d.symmetry.crystal_system if d.symmetry else None,
                "energy_above_hull": d.energy_above_hull,
                "formation_energy_per_atom": d.formation_energy_per_atom,
                "density": d.density,
                "volume": d.volume,
                "nsites": d.nsites,
            }
        )
    return rows


def is_valid_ternary_alloy(formula: str, elem_a: str, elem_b: str, elem_c: str) -> bool:
    """
    checks if the formula contains only the specified elements and that the
    sum of the amounts of elem_a and elem_b equals the amount of elem_c to ensure
    its a valid ternary composition
    """
    comp = Composition(formula)
    el_dict = comp.get_el_amt_dict()
    return el_dict.get(elem_a, 0) + el_dict.get(elem_b, 0) == el_dict.get(elem_c, 0)


def pull_binary_family(api_key):
    all_rows = []
    with MPRester(api_key) as mpr:
        for elem_a, elem_b in ALLOWED_PAIRS:
            chemsys = f"{elem_a}-{elem_b}"
            print(f"Querying {chemsys} ...")

            rows = _query_chemsys(mpr, chemsys, chemsys)
            all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def pull_ternary_family(api_key):
    all_rows = []
    with MPRester(api_key) as mpr:
        for elem_a, elem_b, elem_c in ALL_TERNARY_COMBOS:
            chemsys = f"{elem_a}-{elem_b}-{elem_c}"
            print(f"Querying {chemsys} ...")

            rows = _query_chemsys(mpr, chemsys, chemsys)
            valid_rows = [
                row
                for row in rows
                if is_valid_ternary_alloy(row["formula"], elem_a, elem_b, elem_c)
            ]

            print(f"  {len(valid_rows)}/{len(rows)} passed stoichiometry validation")
            all_rows.extend(valid_rows)

    return pd.DataFrame(all_rows)


def filter_stable(df: pd.DataFrame, ehull_cutoff: float = 0.05) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only compounds with energy_above_hull < 0.05 eV/atom
    for thermodynamically stable or nearly stable compounds.
    """
    before = len(df)
    df_filtered = df[df["energy_above_hull"] <= ehull_cutoff].copy()
    print(
        f"Stability filter: kept {len(df_filtered)}/{before} compounds "
        f"(energy_above_hull <= {ehull_cutoff} eV/atom)"
    )
    return df_filtered

def deduplicate_polymorphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate polymorphs by keeping only the most stable structure
    (lowest energy_above_hull) for each unique formula.
    """
    before = len(df)
    df_sorted = df.sort_values("energy_above_hull")
    df_dedup = df_sorted.drop_duplicates(subset="formula", keep="first")
    print(f"Polymorph dedup: kept {len(df_dedup)}/{before} compounds "
          f"(one most-stable structure per unique formula)")
    return df_dedup

def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
        )

    print("Pulling binary compounds")
    df_binary = pull_binary_family(api_key)
    print(f"Total binary entries pulled: {len(df_binary)}")

    print("\nPulling ternary compounds")
    df_ternary = pull_ternary_family(api_key)
    print(f"Total ternary entries pulled: {len(df_ternary)}")

    df_all = pd.concat([df_binary, df_ternary])
    print(f"\nTotal combined entries: {len(df_all)}")

    df_stable = filter_stable(df_all)

    df_stable = deduplicate_polymorphs(df_stable)

    print("\nCompounds per element system (post-stability-filter):")
    print(df_stable["elements"].value_counts())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_stable.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_stable)} compounds to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()