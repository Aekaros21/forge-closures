"""Versioned SpaRTA constitutive relations verified against primary sources.

The final journal models are NOT the longer arXiv-v2 models in the historical
``library`` module.  Keep that library unchanged so old run identities remain
auditable.  Every constructor here returns an ordinary CandidateSpec already
translated to the OpenFOAM solver's rotation convention: do NOT pass its result
through ``library.to_solver_convention`` a second time.

These specs identify the algebraic channels only.  They do not configure SST
transport, source insertion, realizability projection, or relaxation.  A driver
must record those choices separately; successful construction is not CFD
validation.  Finite 1e30 bounds make coefficient/source caps effectively inactive
at CFD-scale states without writing infinities to OpenFOAM dictionaries.
"""

from __future__ import annotations

from .spec import CandidateSpec, Term

EFFECTIVELY_UNBOUNDED = 1.0e30
JOURNAL_URL = "https://doi.org/10.1007/s10494-019-00089-x"
PREPRINT_URL = "https://arxiv.org/pdf/1905.07510v2"


def _spec(name: str, bdelta: tuple[Term, ...], r_coefficient: str) -> CandidateSpec:
    result = CandidateSpec(
        name=name,
        bdelta=bdelta,
        rsource=(Term("T1", r_coefficient),),
        gmax=EFFECTIVELY_UNBOUNDED,
        r_max_factor=EFFECTIVELY_UNBOUNDED,
    )
    result.validate(search_policy=False)
    return result


def journal_m1_spec() -> CandidateSpec:
    """Schmelzer et al., FTaC 104 (2020), Eq. (22): bDelta=0, bR=.39 T1."""
    return _spec("sparta_schmelzer2020_journal_m1", (), "0.39")


def journal_m2_spec() -> CandidateSpec:
    """Final Eq. (23); paper +4.09 T2 becomes solver -4.09 T2 once."""
    return _spec(
        "sparta_schmelzer2020_journal_m2",
        (Term("T1", "0.1"), Term("T2", "0.0-4.09")),
        "1.39",
    )


def journal_m3_spec() -> CandidateSpec:
    """Final Eq. (24): bDelta=0, bR=.93 T1; R=1.86 k (Sdim:Sdim)/omega."""
    return _spec("sparta_schmelzer2020_journal_m3", (), "0.93")


def preprint_m3_spec(*, xi: float = 1.0) -> CandidateSpec:
    """ArXiv 1905.07510v2 Eq. (27), with the optional Sec. 3.3 bDelta rescue.

    ``xi=1`` preserves the printed polynomial; ``xi=.1`` scales bDelta ONLY.
    Neither option scales bR=.93 T1.  The printed T1 term is 1.251*I1**2,
    not the cubic power transcribed in the historical library.  Invariants use
    I2=tr(Omega_hat**2)<=0.  The solver has W_hat=-Omega_hat, so only the
    T2 coefficient among these four tensors changes sign.
    """
    if xi not in (0.1, 1.0):
        raise ValueError("The source specifies xi=1.0 or the bDelta-only rescue xi=0.1")
    paper_coefficients = (
        ("T1", "0.11*I1*I2+0.27*I1*I2*I2-0.13*I1*I2*I2*I2"
         "+0.07*I1*I2*I2*I2*I2+17.48*I1+0.01*I1*I1*I2"
         "+1.251*I1*I1+3.67*I2+7.52*I2*I2-0.3"),
        ("T2", "0.17*I1*I2*I2-0.16*I1*I2*I2*I2-36.25*I1"
         "-2.39*I1*I1+19.22*I2+7.04"),
        ("T3", "0.0-0.22*I1*I1+1.8*I2+0.07*I2*I2+2.65"),
        ("T4", "0.2*I1*I1-5.23*I2-2.93"),
    )
    terms = []
    for tensor, coefficient in paper_coefficients:
        if tensor == "T2":
            coefficient = f"0.0-({coefficient})"
        terms.append(Term(tensor, f"{xi:g}*({coefficient})"))
    suffix = "xi1" if xi == 1.0 else "xi0p1"
    return _spec(f"sparta_schmelzer_arxiv_v2_m3_{suffix}", tuple(terms), "0.93")


def closure_metadata(spec: CandidateSpec) -> dict[str, object]:
    """Return JSON-compatible provenance for an unmodified constructor result.

    Reject altered specs rather than letting an unchanged display name assign
    primary-source provenance to different coefficients or clipping settings.
    """
    definitions = (
        (journal_m1_spec(), "journal_2020", "22", 1.0),
        (journal_m2_spec(), "journal_2020", "23", 1.0),
        (journal_m3_spec(), "journal_2020", "24", 1.0),
        (preprint_m3_spec(xi=1.0), "arxiv_1905.07510v2", "27", 1.0),
        (preprint_m3_spec(xi=0.1), "arxiv_1905.07510v2", "27", 0.1),
    )
    for canonical, version, equation, xi in definitions:
        if spec == canonical:
            journal = version == "journal_2020"
            return {
                "model_name": spec.name,
                "authors": ["Martin Schmelzer", "Richard P. Dwight", "Paola Cinnella"],
                "title": "Discovery of Algebraic Reynolds-Stress Models Using Sparse Symbolic Regression",
                "source_version": version,
                "equation": equation,
                "source_url": JOURNAL_URL if journal else PREPRINT_URL,
                "time_scale": "1/omega",
                "strain": "Sdim=(gradU+gradU.T)/2; Shat=Sdim/omega (incompressible)",
                "rotation": "solver What=-paper Omega_hat; T2 coefficient translated once",
                "invariants": "I1=tr(Shat^2); I2=tr(What^2)<=0",
                "r_definition": "R=2*k*bR:Sdim; add +R to k and +gamma*R/nut to omega",
                "xi_bdelta": xi,
                "xi_rsource": 1.0,
                "gmax": spec.gmax,
                "r_max_factor": spec.r_max_factor,
                "transport_configuration_in_spec": False,
                "normalization_note": "No k/epsilon time scale, extra betaStar factor, or SST-limited tau",
            }
    raise ValueError("Spec is not an unmodified, versioned SpaRTA constructor result")
