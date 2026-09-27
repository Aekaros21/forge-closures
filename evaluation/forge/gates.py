"""Cheap checks applied to every structure and every coefficient trial."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

from tedp import candidate, tier0, tier1
from tedp.spec import CandidateSpec, spec_hash
from .core import atomic_json, fingerprint


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Gate:
    def __init__(self, *, samples: int = 8192, seed: int = 1729,
                 channel: bool = True, homogeneous: bool = True, output: Path | None = None):
        if samples < 128:
            raise ValueError("At least 128 sampled states are required")
        self.states = candidate.sample_states(samples, seed)
        self.channel, self.homogeneous, self.output = channel, homogeneous, output
        self.baseline = tier1.solve_channel1d(None) if channel else None
        if self.baseline is not None and not self.baseline.converged:
            raise RuntimeError("SST 1D channel gate did not converge")
        self.homogeneous_baseline = tier1.homogeneous_dns_score(None) if homogeneous else None
        if self.homogeneous_baseline is not None and not self.homogeneous_baseline.ok:
            raise RuntimeError("SST homogeneous reference could not be evaluated")
        from tedp.features import FEATURE_VERSION
        # The sampled states carry the V3 physical-state inputs (drawn after the legacy draws, so
        # legacy expressions see the same states); the policy names that state extension.
        self.policy = {"samples": samples, "seed": seed, "channel": channel,
                       "homogeneous": homogeneous, "grammar": "audited-strain-rotation-v1+" + FEATURE_VERSION,
                       "novelty": "reported separately; never a physics substitute"}
        self.cache = {}

    def __call__(self, spec: CandidateSpec) -> dict:
        identity = fingerprint({"spec": spec_hash(spec), "policy": self.policy})
        if identity in self.cache:
            return self.cache[identity]
        cheap = tier0.run_tier0(spec, self.states)
        reasons = list(cheap.reasons)
        detail = {"tensor": finite_json(asdict(cheap))}
        if not reasons and self.channel:
            channel = tier1.channel_gates(spec, self.baseline)
            reasons.extend(channel.reasons)
            detail["channel"] = finite_json(asdict(channel))
        if not reasons and self.homogeneous:
            homogeneous = tier1.homogeneous_dns_score(spec)
            reasons.extend(homogeneous.reasons)
            detail["homogeneous"] = finite_json(asdict(homogeneous))
            detail["homogeneous_sst"] = finite_json(asdict(self.homogeneous_baseline))
        result = {"passed": not reasons, "reasons": reasons, "checks": detail,
                  "policy": self.policy, "identity": identity}
        self.cache[identity] = result
        if self.output:
            atomic_json(self.output / f"{identity}.json", result)
        return result
