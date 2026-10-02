"""QCR2000 (Spalart 2000) on the SST stress as a basis-grammar spec (closures/comparators/qcr2000/qcr2000_comparator.py).

The spec must reproduce an independent NumPy QCR2000 on random velocity-gradient states (any SST
limiter state, omega floor and frame rotation), reduce to SST exactly at C_cr1 = 0, give Spalart's
normal-stress ordering in simple shear, and evaluate identically through the solver's C++ parser.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import qcr2000_comparator as Q
from tedp import spec as spec_mod


@pytest.fixture(scope="module")
def states():
    return Q.sample_states(4000, 7)


def test_spec_round_trips_and_is_not_a_candidate_form():
    spec = Q.qcr2000_spec()
    assert spec_mod.from_json(spec_mod.to_json(spec)) == spec
    assert spec.constants == (0.3,)
    assert spec.gmax == 1.0e30
    assert [t.tensor for t in spec.bdelta] == ["T2"] and spec.rsource == ()
    # comparator-only variable phiDkPk: refused by the search policy, as it must be
    with pytest.raises(spec_mod.SpecError):
        spec.validate(search_policy=True)


def test_matches_independent_qcr2000(states):
    spec = Q.qcr2000_spec()
    ref, ours, rel, _, _ = Q.compare(spec, states)
    nonzero = np.sqrt(np.einsum("nij,nij->n", ref, ref)) > 0
    assert nonzero.sum() > 3000
    assert rel.max() < 1e-9


def test_zero_coefficient_is_exactly_sst(states):
    spec = Q.qcr2000_spec(0.0)
    stress, _, _ = Q.spec_stress(spec, states["grad_u"], states["k"], states["nut"], states["omega"],
                                 states["omega_min"], states["frame_omega"])
    assert np.all(stress == 0.0)


def test_simple_shear_ordering():
    assert Q.simple_shear_check(Q.qcr2000_spec())["passed"]


def test_added_stress_is_traceless_and_does_no_work(states):
    spec = Q.qcr2000_spec()
    _, ours, _, _, _ = Q.compare(spec, states)
    _, _, s_hat = Q.solver_variables(states["grad_u"], states["k"], states["nut"], states["omega"],
                                     states["omega_min"], states["frame_omega"])
    scale = np.sqrt(np.einsum("nij,nij->n", ours, ours)) + 1e-300
    assert np.max(np.abs(np.trace(ours, axis1=1, axis2=2)) / scale) < 1e-12
    work = np.abs(np.einsum("nij,nij->n", ours, s_hat))
    assert np.max(work / (scale * np.sqrt(np.einsum("nij,nij->n", s_hat, s_hat)) + 1e-300)) < 1e-12


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_cpp_parser_agrees(states):
    spec = Q.qcr2000_spec()
    _, _, _, variables, g2 = Q.compare(spec, states)
    cpp = Q.cpp_evaluate(Q.deployed_expression(spec), variables)
    g2 = np.broadcast_to(g2, cpp.shape)
    assert np.max(np.abs(cpp - g2) / np.maximum(np.abs(g2), 1e-300)) < 1e-12
