"""Separate component preservation, family accuracy and useful improvement.

Development temporal sensitivity is not a full validation uncertainty budget.
Release assessment must additionally require reference/discretization coverage.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Any
import numpy as np
from .admission import admission, is_airfoil_control

FAILURE_OBJECTIVE = 1.0e12


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def metric_rules(case: dict, contract: dict) -> dict[str, dict]:
    definitions = case.get("metrics", {})
    if isinstance(definitions, list):
        definitions = {m["name"]: m for m in definitions}
    defaults = contract.get("defaults", {})
    overrides = contract.get("cases", {}).get(case["id"], {})
    result = {}
    for name, definition in definitions.items():
        if not isinstance(definition, dict):
            definition = {}
        if name == "composite" or definition.get("diagnostic_only", False):
            continue
        result[name] = {**defaults, "primary": definition.get("objective", True), **definition,
                        **contract.get("metrics", {}).get(name, {}), **overrides.get(name, {})}
        if is_airfoil_control(case):
            result[name]['primary'] = False
    return result


def assess(case_definitions: list[dict], results: dict, baselines: dict,
           contract: dict, *, release: bool = False) -> dict:
    reasons, rows, families = [], {}, {}
    worst = 0.0
    for case in case_definitions:
        cid = case["id"]
        candidate, stock = results.get(cid), baselines.get(cid)
        if not candidate or not stock:
            reasons.append(f"{cid}: missing candidate or matched SST")
            continue
        candidate_admission, stock_admission = admission(candidate, case), admission(stock, case)
        if not candidate_admission['admitted']:
            reasons.append(f"{cid}: candidate lacks admission for its declared use")
        if not stock_admission['admitted']:
            reasons.append(f"{cid}: SST lacks admission for its declared use")
        if release and not (candidate_admission['fully_qualified'] and stock_admission['fully_qualified']):
            reasons.append(f"{cid}: scoped surface control does not establish release qualification")
        # Library can differ (stock has no correction), everything affecting
        # the physical/numerical comparison must share the paired identity.
        if candidate.get("pairing_hash") != stock.get("pairing_hash"):
            reasons.append(f"{cid}: unmatched experiment definitions")
        if release and not case.get("release_qualified", False):
            reasons.append(f"{cid}: reference/mesh uncertainty not qualified for release")
        primary, weights, gains = [], [], []
        metrics = {}
        for name, rule in metric_rules(case, contract).items():
            if not rule.get("required", True):
                continue
            kind = rule.get("kind", "error")
            field = "observables" if kind in ("departure", "observable_departure", "profile_departure") else "errors"
            a = candidate.get(field, {}).get(name)
            b = stock.get(field, {}).get(name)
            is_profile = kind == "profile_departure"
            profile_change = None
            if is_profile:
                try:
                    aa,bb = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
                    valid = aa.ndim == bb.ndim == 1 and aa.size > 0 and aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
                    if valid:
                        profile_change = float(np.max(np.abs(aa-bb)))
                        a,b = float(np.max(np.abs(aa))),float(np.max(np.abs(bb)))
                except (ValueError, TypeError):
                    valid = False
            else:
                valid = _finite(a) and _finite(b)
            if not valid:
                reasons.append(f"{cid}/{name}: missing or non-finite required metric")
                continue
            floor = float(rule.get("floor", 1e-6))
            if floor <= 0 or not math.isfinite(floor):
                raise ValueError(f"{cid}/{name}: invalid normalization floor")
            allowance = float(rule.get("allowance_abs", 0)) + float(rule.get("allowance_rel", .02)) * abs(b)
            if allowance < 0 or not math.isfinite(allowance):
                raise ValueError(f"{cid}/{name}: invalid engineering allowance")
            ua = candidate.get("uncertainty", {}).get(name)
            ub = stock.get("uncertainty", {}).get(name)
            if not _finite(ua) or not _finite(ub) or min(ua, ub) < 0:
                reasons.append(f"{cid}/{name}: missing paired numerical sensitivity")
                continue
            # Summed absolute sensitivity envelopes are conservative, not a
            # fabricated Gaussian confidence interval or independence claim.
            same_experiment = bool(candidate.get("experiment_hash")) and candidate.get("experiment_hash") == stock.get("experiment_hash")
            uncertainty = 0.0 if same_experiment else ua + ub
            change = profile_change if is_profile else abs(a-b) if field == "observables" else a-b
            upper = change + uncertainty
            passed = upper <= allowance + 1e-14
            violation = 0.0 if passed else max(0.0, (upper - allowance) / max(floor, allowance))
            worst = max(worst, violation)
            metrics[name] = {"candidate": a, "sst": b, "change": change,
                             "sensitivity_bound": uncertainty, "allowance": allowance,
                             "upper_change": upper, "passed": passed, "violation": violation}
            if not passed:
                reasons.append(f"{cid}/{name}: preservation exceeded")
            if field == "errors" and rule.get("primary", True):
                weight = float(rule.get("weight", 1))
                if weight <= 0 or not math.isfinite(weight):
                    raise ValueError("Primary metric weights must be positive")
                primary.append(a/max(abs(b), floor))
                weights.append(weight)
                gains.append((b-a-uncertainty)/max(abs(b), floor))
        if not metrics:
            reasons.append(f"{cid}: no qualified preservation metrics")
        normalized_error = sum(x*w for x,w in zip(primary,weights))/sum(weights) if weights else None
        lower_gain = sum(x*w for x,w in zip(gains,weights))/sum(weights) if weights else None
        rows[cid] = {"metrics": metrics, "normalized_error": normalized_error,
                     "lower_gain": lower_gain, "family": case["family"],
                     "group": case.get("group", case["family"]),
                     "candidate_admission": candidate_admission, "sst_admission": stock_admission}
        if normalized_error is not None:
            families.setdefault(case["family"], {}).setdefault(case.get("group", cid), []).append(normalized_error)
    # Equal families, equal physical groups within each family; Reynolds and
    # mesh sweeps cannot dominate simply by containing more rows.
    family_errors = {f: mean(mean(v) for v in groups.values()) for f,groups in families.items()}
    protection_only = bool(case_definitions) and all(c.get('role') == 'protection' for c in case_definitions)
    if not family_errors and not protection_only:
        reasons.append("No primary accuracy metrics")
    feasible = not reasons
    objective = mean(family_errors.values()) if family_errors else 0. if protection_only else FAILURE_OBJECTIVE
    return {"feasible": feasible, "max_violation": worst if feasible or rows else FAILURE_OBJECTIVE,
            "family_errors": family_errors, "objective": objective,
            "reasons": reasons, "cases": rows,
            "accuracy_scope": "protection_only_no_accuracy_objective" if not family_errors and protection_only else "primary_reference_errors",
            "uncertainty_scope": "temporal_sensitivity; reference and discretization not certified",
            "release": release}


def improvement_verdict(assessment: dict, contract: dict) -> dict:
    """Use a baseline-selected eligibility list; never choose it from new gains."""
    eligible = contract.get("improvement", {}).get("eligible_cases", [])
    if not eligible:
        return {"passed": False, "status": "unresolved", "reasons": ["SST eligibility not frozen"]}
    minimum = float(contract["improvement"].get("minimum_gain", .10))
    fraction = float(contract["improvement"].get("case_fraction", .75))
    median_target = float(contract["improvement"].get("median_gain", .30))
    families, reasons = {}, []
    for cid in eligible:
        row = assessment["cases"].get(cid)
        if row is None or row["lower_gain"] is None:
            reasons.append(f"{cid}: missing required gain")
            continue
        families.setdefault(row["family"], []).append(row["lower_gain"])
    weighted = []
    for family, gains in families.items():
        required = math.ceil(fraction*len(gains))
        if sum(g >= minimum for g in gains) < required:
            reasons.append(f"{family}: insufficient cases meet minimum benefit")
        weighted.extend((g, 1/(len(families)*len(gains))) for g in gains)
    cumulative, median = 0.0, None
    for value, weight in sorted(weighted):
        cumulative += weight
        if cumulative >= .5 - 1e-12:
            median = value
            break
    if median is None or median < median_target:
        reasons.append("Family-balanced median benefit target unmet")
    if not assessment["feasible"]:
        reasons.append("Preservation or qualification failed")
    return {"passed": not reasons, "status": "passed" if not reasons else "failed",
            "reasons": reasons, "weighted_median_lower_gain": median}
