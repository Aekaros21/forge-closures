"""tedp command line: the only sanctioned entry points for phase work.

    tedp smoke            OpenFOAM smoke battery (recovery + live channel)
    tedp verify-recovery  empty-spec kOmegaSSTBasis == stock kOmegaSST
    tedp index            build/refresh the data index (tripwire enforced)
    tedp repro            Phase 0 reproduction table          (needs data)
    tedp noise            Phase 0 noise floor                 (needs data)
    tedp freeze-scoring   one-shot Phase-1 scoring/grammar freeze
    tedp ceiling          Phase 2 frozen-RANS + regression    (needs data)
    tedp search           Phase 3 autonomous search           (post-freeze)
    tedp holdout          Tier 3 — refuses while /srv/tedp-holdout is sealed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOLDOUT = Path("/srv/tedp-holdout")


def cmd_smoke(_args) -> int:
    return subprocess.run(
        ["bash", str(REPO / "tests" / "foam" / "run_foam_smoke.sh")]
    ).returncode


def cmd_verify_recovery(_args) -> int:
    # the smoke battery includes the exact-recovery assertion
    return cmd_smoke(_args)


def cmd_index(_args) -> int:
    from .data import fetch

    index = fetch.build_index()
    print(f"indexed {len(index.cases)} cases in {len(index.families)} families")
    for fam, cases in sorted(index.families.items()):
        print(f"  {fam:12s} {len(cases)} case(s)")
    return 0


def cmd_freeze_scoring(args) -> int:
    from . import evaldb

    payload = evaldb.write_freeze(note=args.note)
    print("scoring + grammar frozen:")
    for rel, sha in payload["hashes"].items():
        print(f"  {sha[:12]}  {rel}")
    return 0


def cmd_holdout(_args) -> int:
    import os

    if not HOLDOUT.exists() or not os.access(HOLDOUT, os.R_OK):
        print(
            "holdout is sealed (this is the correct state before Checkpoint 3).\n"
            "Only after FREEZE.json exists and the human runs\n"
            "  sudo bash scripts/unseal_holdout.sh\n"
            "does this command proceed.",
            file=sys.stderr,
        )
        return 2
    print("holdout is unsealed — Tier-3 runner not implemented yet (Phase 4)")
    return 1


def _todo(phase: str):
    def run(_args) -> int:
        print(f"{phase}: not implemented yet", file=sys.stderr)
        return 1

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tedp")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke").set_defaults(func=cmd_smoke)
    sub.add_parser("verify-recovery").set_defaults(func=cmd_verify_recovery)
    sub.add_parser("index").set_defaults(func=cmd_index)
    p = sub.add_parser("freeze-scoring")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_freeze_scoring)
    sub.add_parser("repro").set_defaults(func=_todo("repro"))
    sub.add_parser("noise").set_defaults(func=_todo("noise"))
    sub.add_parser("ceiling").set_defaults(func=_todo("ceiling"))
    sub.add_parser("search").set_defaults(func=_todo("search"))
    sub.add_parser("holdout").set_defaults(func=cmd_holdout)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
