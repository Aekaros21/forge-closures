"""McConkey/Yee/Lien 2021 dataset access.

Layout under data/mcconkey/:
    foam/komegasst/<family>/<case>/   complete, already-run OpenFOAM cases
                                      (0/, constant/, system/, <tFinal>/)
    komegasst.csv                     SST fields at cell centres, per Case
    REF.csv                           DNS/LES labels (U, tau, k, b, gradU,
                                      divtau, yplus, Uplus) per Case

Case roles follow the experimental plan. The develop/precursor cases exist
for inlet development and are never scored. The duct family never reaches
this directory (sealed at fetch time).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MC = REPO / "data" / "mcconkey"
FOAM = MC / "foam" / "komegasst"   # holdout_duct sits beside the case families
CACHE = MC / "cache"

CALIBRATION = {
    "case_0p5": ("pehill", "case_0p5"),
    "case_1p0": ("pehill", "case_1p0"),
    "case_1p5": ("pehill", "case_1p5"),
    "convdiv12600": ("convdiv", "convdiv12600"),
    "convdiv20580": ("convdiv", "convdiv20580"),
    "cbfs13700": ("cbfs", "cbfs13700"),
}
VALIDATION = {
    "case_0p8": ("pehill", "case_0p8"),
    "case_1p2": ("pehill", "case_1p2"),
    "h20": ("bump", "h20"),
    "h26": ("bump", "h26"),
    "h31": ("bump", "h31"),
    "h38": ("bump", "h38"),
    "h42": ("bump", "h42"),
}
GATES = {
    "fp_3630": ("fp", "flatplate"),
}
# Phase-4 holdout: the square-duct RANS templates staged from the sealed
# dataset (mesh + initial conditions only, so runs start cold exactly as the
# development cases do). Scored against the sealed DNS by tedp.holdout_cases
# .duct_scoring, not by the development REF tables.
HOLDOUT_DUCT = {
    "squareDuct_Re_1100": ("holdout_duct", "squareDuct_Re_1100"),
    "squareDuct_Re_2000": ("holdout_duct", "squareDuct_Re_2000"),
    "squareDuct_Re_3500": ("holdout_duct", "squareDuct_Re_3500"),
}
ALL_CASES = {**CALIBRATION, **VALIDATION, **GATES, **HOLDOUT_DUCT}
# Post-search verification only: the other thirteen Reynolds numbers of the same square-duct DNS
# set (held-out conditions of an exposed lineage, not blind tests), fetched from the published
# dataset by scripts/verify_fetch_duct_kaggle.py. Kept out of ALL_CASES so development tooling
# that enumerates the dataset is unchanged.
VERIFICATION_DUCT = {
    f"squareDuct_Re_{re}": ("holdout_duct", f"squareDuct_Re_{re}")
    for re in (1150, 1250, 1300, 1350, 1400, 1500, 1600, 1800, 2205, 2400, 2600, 2900, 3200)
}


def duct_reference_file(case_id: str) -> Path:
    """Cached DNS reference arrays for one holdout duct case."""
    return CACHE / f"holdout_duct_{case_id}.npz"


def case_dir(case_id: str) -> Path:
    fam, sub = {**ALL_CASES, **VERIFICATION_DUCT}[case_id]
    d = FOAM / fam / sub
    if not d.is_dir():
        raise FileNotFoundError(d)
    return d


def final_time(case_id: str) -> str:
    d = case_dir(case_id)
    times = sorted(
        (int(p.name) for p in d.iterdir() if p.name.isdigit()), reverse=True
    )
    if not times:
        raise FileNotFoundError(f"no result time in {d}")
    return str(times[0])


_csv_case_names: dict[str, list[str]] | None = None


def load_case_table(case_id: str, source: str = "REF") -> dict[str, np.ndarray]:
    """Columns for one case from REF.csv / komegasst.csv, cached as npz.

    The CSV Case column uses its own naming; `case_id` here IS that CSV name
    (the keys of CALIBRATION/VALIDATION/GATES are chosen to match it)."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"{source}_{case_id}.npz"
    if cache_file.exists():
        return dict(np.load(cache_file))

    import pandas as pd

    csv = MC / f"{source if source == 'REF' else 'komegasst'}.csv"
    df = pd.read_csv(csv)
    if case_id not in set(df["Case"]):
        raise KeyError(
            f"{case_id!r} not in {csv.name}; available: {sorted(set(df['Case']))}"
        )
    sub = df[df["Case"] == case_id].drop(columns=["Case"])
    arrays = {c: sub[c].to_numpy() for c in sub.columns if sub[c].dtype != object}
    np.savez_compressed(cache_file, **arrays)
    return arrays


def csv_case_names() -> list[str]:
    import pandas as pd

    df = pd.read_csv(MC / "REF.csv", usecols=["Case"])
    return sorted(set(df["Case"]))
