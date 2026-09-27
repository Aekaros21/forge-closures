"""Grouped development catalogue; asset presence never certifies a release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ADAPTERS = {"mcconkey", "duct", "rotation", "plate", "bump", "naca", "hump", "faith", "rectduct", "ahmed", "diffuser3d", "wingbody"}
ROLES = {"calibration", "adaptive_validation", "protection", "final", "verification"}


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def validate_manifest(manifest: dict, root: str | Path) -> dict:
    """Reject leakage/malformed declarations; report missing/weak data separately.

    ``ready`` means staged inputs are available for development. It deliberately
    does not mean independent final tests or validated CFD capability exist.
    """
    root = Path(root).resolve()
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported catalogue schema")
    entries = manifest.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("A nonempty cases list is required")
    ids, groups, missing, weak = set(), {}, [], []
    required_keys = {"id", "family", "group", "role", "adapter", "adapter_config",
                     "required", "reference", "metrics", "protocol"}
    for case in entries:
        if required_keys - case.keys():
            raise ValueError(f"Missing case keys: {required_keys - case.keys()}")
        cid = case["id"]
        if cid in ids or not isinstance(cid, str) or not cid:
            raise ValueError(f"Duplicate/invalid case id: {cid}")
        ids.add(cid)
        if case["adapter"] not in ADAPTERS or case["role"] not in ROLES:
            raise ValueError(f"Unsupported adapter/role for {cid}")
        if not case["group"] or not case["metrics"]:
            raise ValueError(f"Missing physical group or metrics for {cid}")
        groups.setdefault(case["group"], set()).add(case["role"])
        if case["role"] == "final" and case.get("exposure", "exposed") != "locked_unseen":
            raise ValueError(f"Exposed case cannot be a final test: {cid}")
        if case["role"] == "final" and not case.get("independence_evidence"):
            raise ValueError(f"Final case lacks physical independence evidence: {cid}")
        # A held-out condition or numerical study from an exposed lineage. It is evaluated after
        # the search and is never a blind test, so it must say which kind it is.
        if case["role"] == "verification" and case.get("exposure") not in ("held_out_condition", "numerical_study"):
            raise ValueError(f"Verification case must declare held_out_condition or numerical_study exposure: {cid}")
        if case["reference"].get("quality") != "qualified_validation":
            weak.append(cid)
        seen_metrics = set()
        for metric in case["metrics"]:
            if not isinstance(metric, dict) or not metric.get("name"):
                raise ValueError(f"Invalid metric declaration in {cid}")
            if metric["name"] in seen_metrics:
                raise ValueError(f"Duplicate metric in {cid}: {metric['name']}")
            seen_metrics.add(metric["name"])
            if metric.get("release_qualified") and not metric.get("qualification_evidence"):
                raise ValueError(f"Unsubstantiated release metric in {cid}")
        for asset in case.get("assets", []):
            p = root / asset["path"]
            if not p.resolve().is_relative_to(root):
                raise ValueError(f"External/symlinked mutable asset for {cid}: {p}")
            if not p.exists():
                missing.append({"case": cid, "path": asset["path"]})
            elif asset.get("sha256") and p.is_file():
                if hashlib.sha256(p.read_bytes()).hexdigest() != asset["sha256"]:
                    raise ValueError(f"Asset hash mismatch: {p}")
        if int(case["protocol"].get("end_time", 0)) <= 0:
            raise ValueError(f"Missing positive endpoint for {cid}")
    for group, roles in groups.items():
        if "final" in roles and len(roles) != 1:
            raise ValueError(f"Final/development group leakage: {group}")
        if {"calibration", "adaptive_validation"} <= roles:
            raise ValueError(f"Calibration/validation physical group leakage: {group}")
    final = [c["id"] for c in entries if c["role"] == "final"]
    return {"ready": not missing, "case_count": len(entries), "physical_group_count": len(groups),
            "missing_assets": missing, "reference_qualification_pending": weak,
            "final_test_cases": final, "independent_release_ready": False,
            "release_blockers": (["No independently curated locked final-test groups"] if not final else [])
                + ["Per-observable reference/numerical qualification and tolerances remain required"],
            "scope": manifest.get("scope")}
