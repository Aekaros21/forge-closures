"""Convergence evidence for the TMR verification cases.

A run is not reportable because its residual looks small. OpenFOAM normalises
residuals by a factor that grows with the domain, and the TMR aerofoil domain is
703 chords across, so its residuals read two orders of magnitude lower than the
bump's at the same state of convergence. Both cases produced grid sequences that
moved away from the reference under refinement, and in the bump case, where the
run was continued, that was entirely the iteration cut.

So convergence is established here from the *reported quantity itself*: every
case records the integrated force coefficient at every iteration, and a run
counts as converged only if that coefficient has stopped moving. Runs without
that history are not converged; they are unknown, which is treated the same way.

``control_dict`` installs the history. ``status`` reads it back. Nothing in the
reporting path should print a number for a case whose status is not "converged".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The test is not "is it still moving slowly" but "how far is it still going to
# move". A slowly converging run has a small per-block drift while remaining far
# from its own answer: bump 705x321 looked settled at 6000 iterations and its
# viscous drag then moved 3.4% on the way to 24000. So the criterion estimates the
# REMAINING change by summing the fitted geometric decay of the block-mean
# increments, and only skips that when the block means have visibly arrived.
# Set from what is being measured, not picked. The smallest effect reported from
# these cases is the headline-minus-stock drag difference, 0.00023 on 0.0080, or
# 2.9% of the value. A 0.1% bound on remaining change is 3% of that smallest
# signal, so numerical drift stays well below the effect. Tightening beyond this
# buys nothing: it moves the fifth significant figure and costs hours per case.
DEFAULT_TOL = 1.0e-3          # estimated remaining change, 0.1% of the value
# Convergence is a statement about the asymptotic region, so only the last half
# of the history is judged. Including the startup transient puts it inside the
# first block, where it dominates the first increment and describes the
# transient rather than the approach to the answer.
ASYMPTOTIC = 0.5
BLOCKS = 10                   # block means across the asymptotic region
MIN_PER_BLOCK = 20            # iterations, below which a block mean is noise
# Iterations discarded after each restart. A case runs in chunks that checkpoint
# and resubmit, and a restart re-establishes the solver's state from fields
# written at finite precision, so the force coefficient re-settles over a few
# hundred iterations. Those samples are not part of the converged trajectory:
# bump 353x161 restarted at 22000 and swung until 22344, naca449 a2 restarted at
# 40000 and swung until 40703, and in both the last 2000 iterations then held to
# better than 0.01%. Judged with the transients in, both read as wandering by
# half a percent. 1000 covers the longest observed with margin.
RESTART_SETTLE = 1000


def control_dict(end_time: int, wall_patch: str, a_ref: float,
                 application: str = "simpleFoam") -> str:
    """controlDict with wall shear output and a per-iteration force history."""
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}


application     {application};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {end_time};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    wallShear
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         ({wall_patch});
        writeControl    writeTime;
    }}
    forceHistory
    {{
        type            forceCoeffs;
        libs            (forces);
        patches         ({wall_patch});
        rho             rhoInf;
        rhoInf          1;
        magUInf         1;
        lRef            1;
        Aref            {a_ref};
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        CofR            (0 0 0);
        pitchAxis       (0 0 1);
        writeControl    timeStep;
        writeInterval   1;
        writeFields     no;
        log             no;
    }}
}}
"""


@dataclass
class Status:
    converged: bool
    reason: str
    drift: float | None = None
    iterations: int | None = None
    final: float | None = None

    def __str__(self) -> str:
        if self.converged:
            return (f"converged ({self.drift:.2e} of the value estimated still to "
                    f"move, at iteration {self.iterations})")
        return f"NOT CONVERGED: {self.reason}"


def _read_history(case: Path,
                  alpha_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Iteration and drag coefficient from every forceCoeffs history in the case.

    ``forceCoeffs`` is configured with dragDir (1 0 0), so its Cd column is the
    AXIAL force. Incidence is imposed by rotating the freestream, so at alpha
    that column is not the drag being reported: it is Cd cos a - Cl sin a, a
    small residual of two much larger opposing terms. At 2 degrees on the
    aerofoil it is 0.00064 where the drag itself is 0.0064, so judging it holds
    the run to roughly ten times the tolerance actually being asked for, on a
    quantity nobody reports. ``alpha_deg`` resolves the columns back along the
    freestream so the criterion certifies the number that is shown.
    """
    files = sorted(Path(case).glob("postProcessing/forceHistory/*/coefficient*.dat"))
    if not files:
        return None
    times, cds = [], []
    for f in files:
        # the directory is named for the time the chunk started; anything but the
        # first is a restart, whose first iterations are the solver re-settling
        try:
            start = float(f.parent.name)
        except ValueError:
            start = 0.0
        floor = start + RESTART_SETTLE if start > 0 else 0.0
        for line in f.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                tv = float(parts[0])
                cv = float(parts[1])
                if alpha_deg:
                    # columns: Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) Cm...
                    a = math.radians(alpha_deg)
                    cv = cv * math.cos(a) + float(parts[4]) * math.sin(a)
            except (ValueError, IndexError):
                continue
            if tv <= floor:
                continue
            times.append(tv)
            cds.append(cv)
    if len(times) < 10:
        return None
    o = np.argsort(times)
    t, c = np.asarray(times)[o], np.asarray(cds)[o]
    keep = np.concatenate([[True], np.diff(t) > 0])       # drop restart duplicates
    return t[keep], c[keep]


def _decay(d: np.ndarray) -> tuple[float, float] | None:
    """Amplitude and rate of a decaying sequence of increments, |d_k| = A r^k.

    Fitted by least squares across every increment. Reading the rate off the
    last two increments instead makes the verdict a ratio of two small noisy
    numbers, which is how an earlier version of this test came to pass one
    705x321 run at 9.2e-4 and fail its partner at 1.2e-3 while the two were
    drifting at 0.040% and 0.039% per 5,000 iterations. Three equally spaced
    segment means give a closed form for what is left, but it divides by the
    difference of two nearly equal movements and explodes exactly where the
    answer matters most; the fit across all increments does not.
    """
    mag = np.abs(d)
    use = mag > 0
    if use.sum() < 4:
        return None
    k = np.arange(d.size)[use].astype(float)
    y = np.log(mag[use])
    # closed-form least squares rather than np.polyfit: polyfit goes through
    # LAPACK, and the cluster venv's numpy segfaults there inside the extraction
    # jobs. The fit is one line of algebra and needs no linear-algebra backend.
    n = k.size
    sk, sy = k.sum(), y.sum()
    denom = n * (k * k).sum() - sk * sk
    if denom == 0.0:
        return None
    slope = (n * (k * y).sum() - sk * sy) / denom
    intercept = (sy - slope * sk) / n
    return float(np.exp(intercept)), float(np.exp(slope))


def status(case: Path, tol: float = DEFAULT_TOL, blocks: int = BLOCKS,
           at_time: int | None = None, alpha_deg: float = 0.0) -> Status:
    """How much further is the reported coefficient going to move?

    Judged over the ASYMPTOTIC REGION only, taken as the last half of the
    history. Averaging across the whole run puts the startup transient inside
    the first window, so the increment it produces describes the transient and
    not the approach to the answer: bump 705x321 opens at 0.00497 against a
    final 0.00358, which left the previous version of this test with two usable
    increments and a verdict that swung on their ratio. It passed the headline
    at 9.2e-4 and failed the stock run at 1.2e-3 when the two were drifting at
    0.040% and 0.039% per 5,000 iterations, a distinction made entirely by
    where each run happened to stop.

    Ten block means are formed over that region. If they hold to better than a
    tenth of the tolerance across it, the run has arrived and nothing is
    extrapolated: fitting a decay to increments that small fits noise, and doing
    so rejected runs whose block means agreed to seven significant figures.
    Otherwise the increments are fitted to a geometric decay and the sum of the
    remaining terms is what is still to move; increments that are not decaying
    mean the run is still going somewhere, and it fails.

    What this does NOT do is bound the remaining change without assuming the
    approach is geometric. No criterion can: a movement that decays slowly
    enough is indistinguishable, over any finite window, from one that has
    stopped. The assumption is stated rather than hidden, and the tenth-of-
    tolerance shortcut is deliberately an order of magnitude clear of the
    threshold so that a slow approach has to be very slow indeed to slip past.

    ``at_time`` truncates the history to the state whose fields are being
    reported. Without it a run that has since progressed further would be
    certified on evidence that does not belong to the numbers being shown.
    """
    hist = _read_history(Path(case), alpha_deg)
    if hist is None:
        return Status(False, "no force history recorded; convergence unknown")
    t, c = hist
    if at_time is not None:
        # The force history is written every iteration but the fields only at
        # write time, so the history runs ahead of the data actually reported.
        keep = t <= at_time
        if keep.sum() < 40:
            return Status(False,
                          f"only {int(keep.sum())} iterations of history at or before "
                          f"the reported time {at_time}; cannot judge that state")
        t, c = t[keep], c[keep]

    n = t.size
    tail = c[n - int(n * ASYMPTOTIC):]
    if tail.size < blocks * MIN_PER_BLOCK:
        return Status(False,
                      f"only {n} iterations recorded, too few to judge an "
                      f"asymptotic region")
    w = tail.size // blocks
    means = np.array([float(tail[i * w:(i + 1) * w].mean()) for i in range(blocks)])
    value = means[-1]
    if abs(value) < 1e-30:
        return Status(False, "force coefficient is zero")
    last_iter = int(t[-1])

    span = float(np.ptp(means)) / abs(value)
    if span < tol / 10.0:
        # The block means hold to a tenth of the tolerance across the whole
        # asymptotic region. There is no movement here worth extrapolating, and
        # fitting a decay to increments this small fits noise: it was rejecting
        # runs whose block means agreed to seven figures because their last
        # increments happened not to shrink.
        return Status(True, "settled: the block means hold to better than a tenth "
                            "of the tolerance across the asymptotic region",
                      span, last_iter, value)

    d = np.diff(means)
    fit = _decay(d)
    if fit is None:
        return Status(False,
                      f"increments are degenerate but the block means still range "
                      f"over {span:.2e} of the value (tolerance {tol:.0e})",
                      span, last_iter, value)
    amp, r = fit
    if r >= 1.0:
        return Status(False,
                      f"block-mean increments are not decaying (rate {r:.2f}); the "
                      f"run is still moving, range {span:.2e} of the value over the "
                      f"asymptotic region",
                      span, last_iter, value)
    frac = float(amp * r ** d.size / (1.0 - r) / abs(value))
    if frac > span:
        # The prediction of what is left exceeds the movement actually seen over
        # the whole asymptotic region, so it rests on the fitted rate rather than
        # on evidence. The extrapolation is 1/(1-r) and blows up as r approaches
        # 1: bump 705x321 stock and headline were drifting at 0.9e-6 and 0.8e-6
        # per 6,000 iterations, physically the same state, and the fit put them
        # 2.5x apart. Refuse rather than certify on an ill-conditioned fit.
        return Status(False,
                      f"estimated {frac:.2e} of the value still to move, more than "
                      f"the {span:.2e} it has moved across the whole asymptotic "
                      f"region; the decay is too slow to bound from this history",
                      frac, last_iter, value)
    if frac > tol:
        return Status(False,
                      f"estimated {frac:.2e} of the value still to move "
                      f"(tolerance {tol:.0e}); needs more iterations",
                      frac, last_iter, value)
    return Status(True, "settled", frac, last_iter, value)
