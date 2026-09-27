#!/usr/bin/env python3
"""Build, run and score several cases, and compare their case errors with the paper.

    python reproduce/run_cases.py [--cases ID ...] [--closures NAME ...] [--cores N] [--rerun] [--tolerance T]

Without --cases, every case that runs on at most four MPI ranks and whose inputs are present is used: the
two-dimensional and small cases of the paper. Without --closures, SST and the invariant and flow-state
corrections are run, and the rotation-limited correction on the rotating channels (elsewhere it equals the
flow-state correction). Runs start as cores become free (--cores, default all). A finished run is reused
unless --rerun is given. Run in an OpenFOAM v2312 environment after building the library
(closures/Allwmake). The table printed at the end is also written to runs/summary.json. With --tolerance,
the script fails if a case error differs from the paper's value by more than T.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reproduce'))
from score_case import score  # noqa: E402


def inputs_present(case):
    paths = [a['path'] for a in case.get('assets', [])] + list(case.get('reference', {}).get('paths', []))
    return [p for p in paths if not (ROOT/p).is_file()]


def ended(log):
    # an OpenFOAM log ends with 'End' (a parallel run adds 'Finalising parallel run' after it)
    return log.is_file() and 'End' in [line.strip() for line in log.read_text(errors='replace').splitlines()[-3:]]


def finished(work):
    if not (work/'forge_case.json').is_file() or not ended(work/'log.simpleFoam'):
        return False
    meta = json.loads((work/'forge_case.json').read_text())
    return all(ended(work/('log.' + ''.join(ch for ch in f if ch.isalnum()))) for f in meta['required_postprocess'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', nargs='+', help='case ids (default: the cases on at most 4 ranks with inputs present)')
    ap.add_argument('--closures', nargs='+', help='closures besides SST (default: invariant flow_state, and '
                    'rotation_limited on the rotating channels)')
    ap.add_argument('--cores', type=int, default=os.cpu_count(), help='cores to use at once (default: all)')
    ap.add_argument('--rerun', action='store_true', help='rerun cases that have already finished')
    ap.add_argument('--tolerance', type=float, help='fail if a case error differs from the paper by more than this')
    a = ap.parse_args()
    defs = {c['id']: c for c in json.loads((ROOT/'reproduce/cases.json').read_text())['cases']}
    if a.cases:
        unknown = [c for c in a.cases if c not in defs]
        if unknown:
            sys.exit(f'unknown case(s): {", ".join(unknown)}; see reproduce/make_case.py --list')
        ids = a.cases
    else:
        ids = [c for c, d in defs.items() if int(d['protocol'].get('nprocs', 1)) <= 4]
    jobs = []
    for cid in ids:
        missing = inputs_present(defs[cid])
        if missing:
            print(f'skip {cid}: {len(missing)} input(s) missing, e.g. {missing[0]} (see data/README.md)')
            continue
        closures = a.closures or (['invariant', 'flow_state'] +
                                  (['rotation_limited'] if defs[cid]['adapter'] == 'rotation' else []))
        jobs += [(cid, m) for m in ['sst'] + [m for m in closures if m != 'sst']]
    todo = [(cid, m) for cid, m in jobs if a.rerun or not finished(ROOT/'runs'/f'{cid}_{m}')]
    if todo and os.environ.get('WM_PROJECT_VERSION') != 'v2312':
        sys.exit('source the etc/bashrc of OpenFOAM v2312 first (and build the library with ./closures/Allwmake)')
    pending = []
    for cid, m in todo:
        made = subprocess.run([sys.executable, str(ROOT/'reproduce/make_case.py'), cid, m, '--force'],
                              capture_output=True, text=True)
        if made.returncode:
            sys.exit(f'could not build {cid} with {m}: {(made.stderr or made.stdout).strip()}')
        pending.append((cid, m, int(defs[cid]['protocol'].get('nprocs', 1))))
    print(f'{len(jobs)} runs, {len(pending)} to run on up to {a.cores} cores')
    running, failed, started = [], [], time.time()
    while pending or running:
        for job in running[:]:
            if job[0].poll() is not None:
                running.remove(job)
                state = 'done' if job[0].returncode == 0 else f'FAILED (see runs/{job[1]}/log.*)'
                if job[0].returncode:
                    failed.append(job[1])
                print(f'{time.time() - started:7.0f} s  {job[1]}: {state}', flush=True)
        used = sum(job[2] for job in running)
        while pending and (used + pending[0][2] <= a.cores or not running):
            cid, m, n = pending.pop(0)
            work = ROOT/'runs'/f'{cid}_{m}'
            running.append((subprocess.Popen([str(work/'Allrun')], stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL), work.name, n))
            used += n
        time.sleep(2)
    rows = []
    for cid, m in jobs:
        if m == 'sst' or f'{cid}_{m}' in failed or f'{cid}_sst' in failed:
            continue
        r = score(ROOT/'runs'/f'{cid}_{m}', ROOT/'runs'/f'{cid}_sst')
        rows.append({k: r[k] for k in ('case', 'closure', 'case_error', 'paper')})
    print(f'\n{"case":24s}{"closure":18s}{"case error":>11s}{"paper":>9s}{"difference":>12s}')
    for r in rows:
        paper = f'{r["paper"]:9.4f}' if r['paper'] is not None else f'{"-":>9s}'
        diff = f'{r["case_error"] - r["paper"]:+12.4f}' if r['paper'] is not None else f'{"-":>12s}'
        print(f'{r["case"]:24s}{r["closure"]:18s}{r["case_error"]:11.4f}{paper}{diff}')
    compared = [abs(r['case_error'] - r['paper']) for r in rows if r['paper'] is not None]
    if compared:
        print(f'largest difference from the paper: {max(compared):.4f} over {len(compared)} case errors')
    (ROOT/'runs').mkdir(exist_ok=True)
    (ROOT/'runs/summary.json').write_text(json.dumps({'runs': rows, 'failed': failed}, indent=1) + '\n')
    if failed:
        sys.exit(f'{len(failed)} run(s) failed: {", ".join(failed)}')
    if a.tolerance is not None and compared and max(compared) > a.tolerance:
        sys.exit(f'a case error differs from the paper by more than {a.tolerance}')


if __name__ == '__main__':
    main()
