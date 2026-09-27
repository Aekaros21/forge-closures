"""Phase 4 (Tier 3): the holdout, run once after Checkpoint 3.

Refuses to do anything while /srv/tedp-holdout is unreadable. Verifies the
manifest before touching a byte. Three flow classes:

  duct     — McConkey squareDuct_Re_{1100,2000,3500}: shipped komegasst cases
             + DNS fields (Pinelli et al. 2010) under duct/foam/DNS/...
  hump     — NASA wall-mounted hump: Greenblatt experiment (.dat) is sealed;
             the TMR grids were NOT captured by the seal fetch -> the mesh
             must be generated (Glauert-Goldschmied profile) before Phase 4.
  rotchan  — AGARD PCH22 (Kristoffersen & Andersson): profiles at
             Ro = 0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.50; plan uses 0.1, 0.5.
             Case: periodic channel + Coriolis fvOption (to be built).

Only the duct path is fully mechanised here (it mirrors tier2 on shipped
cases). hump/rotchan need their case generators — tracked in results/LOG.md.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOLDOUT = Path("/srv/tedp-holdout")
MANIFEST = REPO / "holdout.manifest.sha256"

DUCT_CASES = ("squareDuct_Re_1100", "squareDuct_Re_2000", "squareDuct_Re_3500")
ROTCHAN_RO = (0.10, 0.50)


class HoldoutSealed(RuntimeError):
    pass


def require_unsealed() -> None:
    if not HOLDOUT.exists() or not os.access(HOLDOUT, os.R_OK):
        raise HoldoutSealed(
            "holdout is sealed — correct before Checkpoint 3; the human runs "
            "sudo bash scripts/unseal_holdout.sh exactly once"
        )
    if not (REPO / "results" / "FREEZE.json").exists():
        raise HoldoutSealed("FREEZE.json missing — the candidate must be frozen first")


def verify_manifest() -> int:
    """Every sealed file must match the manifest written at seal time."""
    require_unsealed()
    n = 0
    for line in MANIFEST.read_text().splitlines():
        sha, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        path = HOLDOUT / rel[2:] if rel.startswith("./") else HOLDOUT / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha:
            raise RuntimeError(f"manifest mismatch: {rel}")
        n += 1
    return n


def duct_case_dir(case: str) -> Path:
    require_unsealed()
    return HOLDOUT / "duct" / "foam" / "komegasst" / "squareDuct" / case


def duct_dns_dir(case: str) -> Path:
    require_unsealed()
    return HOLDOUT / "duct" / "foam" / "DNS" / "squareDuct" / case


# The duct runner will reuse tier2.make_candidate_case / evaluate_candidate
# with a holdout-aware case_dir resolver and a DNS-field reader for the
# reference (fluidfoam on the DNS dir), plus the secondary-flow figure
# (in-plane velocity magnitude) the plan asks for. Implemented in Phase 4
# once the sealed field formats can be inspected.
