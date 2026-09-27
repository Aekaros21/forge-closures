"""Index the fetched calibration/validation data.

THE TRIPWIRE: the index refuses to contain anything that looks like holdout
(square duct, NASA hump, rotating channel / PCH22). The real defense is the
root-owned /srv/tedp-holdout seal; this assertion catches accidents like a
mis-run seal script leaving duct cases in data/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"

HOLDOUT_PATTERN = re.compile(r"duct|hump|pch22|rotat", re.IGNORECASE)

# family dirs under data/mcconkey/foam/komegasst/
FAMILIES = {
    "pehill": "periodic hills (Xiao et al. 2020)",
    "bump": "parametric bumps (Matai & Durbin 2019)",
    "convdiv": "converging-diverging channel (Laval & Marquillie)",
    "cbfs": "curved backward-facing step (Bentaleb et al. 2012)",
    "fp": "flat plate (Schlatter & Orlu 2010)",
}


class HoldoutLeak(RuntimeError):
    pass


@dataclass
class DataIndex:
    cases: dict[str, Path] = field(default_factory=dict)      # case id -> dir
    families: dict[str, list[str]] = field(default_factory=dict)
    external: dict[str, Path] = field(default_factory=dict)   # LM5200, AGARD


def build_index(data_dir: Path = DATA) -> DataIndex:
    index = DataIndex()

    for p in sorted(data_dir.rglob("*")):
        rel = str(p.relative_to(data_dir))
        if HOLDOUT_PATTERN.search(rel):
            raise HoldoutLeak(
                f"holdout-looking path inside data/: {rel!r} — the seal "
                "script must quarantine this before any indexing"
            )

    # aggregated label tables can smuggle holdout ROWS past the path check
    # (this actually happened with the squareDuct rows on first ingestion);
    # verify the Case column of every CSV
    import pandas as pd

    for csv in sorted(data_dir.rglob("*.csv")):
        if csv.stat().st_size < 1024 or "cache" in csv.parts:
            continue
        try:
            cases = pd.read_csv(csv, usecols=["Case"])["Case"].astype(str)
        except (ValueError, KeyError):
            continue  # no Case column — not a per-case label table
        bad = sorted(set(cases[cases.str.contains(HOLDOUT_PATTERN, na=False)]))
        if bad:
            raise HoldoutLeak(
                f"holdout rows inside {csv.relative_to(data_dir)}: {bad[:3]}"
            )

    foam = data_dir / "mcconkey" / "foam" / "komegasst"
    if foam.exists():
        for fam in FAMILIES:
            fam_dir = foam / fam
            if not fam_dir.is_dir():
                continue
            for case_dir in sorted(fam_dir.iterdir()):
                if not case_dir.is_dir() or case_dir.name == "writeFields":
                    continue
                index.cases[case_dir.name] = case_dir
                index.families.setdefault(fam, []).append(case_dir.name)

    external = data_dir / "external"
    if external.exists():
        for p in sorted(external.iterdir()):
            index.external[p.name] = p

    return index
