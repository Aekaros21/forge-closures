#!/usr/bin/env python3
"""Turn a fresh SST case written by reproduce/make_case.py into a case of a compared closure (Sec. V of the paper).

    python reproduce/make_case.py <case_id> sst --name <case_id>_<comparator>
    python reproduce/use_comparator.py runs/<case_id>_<comparator> <comparator>

<comparator>  sstrc (SST-RC), earsm (EARSM) or autoturb (AutoTurb)

The case of a compared closure is the SST case of the same flow, byte for byte, except two files, as in the paper's
comparison campaign (evaluation/forge_fixed, prepare_fixed_case): constant/turbulenceProperties selects the closure's
RAS model with its published coefficients (closures/comparators/comparator_models.json) and the frame rotation of the
case, and system/controlDict loads the closure's library after the library of the corrections. The conversion is made
in place, on an SST case that has not been run. Run the case with its Allrun script and score it with score_case.py
against an SST run of the same case. Build the libraries first with ./closures/comparators/Allwmake. QCR2000 runs in
the library of the corrections and needs no conversion: python reproduce/make_case.py <case_id> qcr2000.
"""
import argparse, hashlib, json, os, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))
COMPARATORS = {'sstrc': 'SSTRC', 'earsm': 'EARSM', 'autoturb': 'AutoTurb'}   # name -> label in the records


def comparator(name):
    """The frozen definition of a compared closure, checked against the sources in closures/comparators."""
    from forge_fixed import FixedModel, source_sha256
    models = json.loads((ROOT/'closures/comparators/comparator_models.json').read_text())['models']
    entry = next(m for m in models if m['label'] == COMPARATORS[name])
    sources = ROOT/'closures/comparators'/Path(entry['fixed']['source_dir']).name
    if source_sha256(sources) != entry['fixed']['source_sha256']:
        sys.exit(f'{sources} differs from the sources of the paper (source_sha256 in comparator_models.json)')
    return FixedModel.from_entry(entry)


def library(model):
    """The built library, copied under a name that carries its SHA-256 prefix, as make_case.py does for its own."""
    built = Path(os.environ.get('FOAM_USER_LIBBIN', '~')).expanduser()/model.library
    if not built.is_file():
        sys.exit(f'{model.library} not found: source OpenFOAM v2312, then run ./closures/comparators/Allwmake')
    digest = hashlib.sha256(built.read_bytes()).hexdigest()
    copy = ROOT/'build'/f'{Path(model.library).stem}-{digest[:12]}.so'
    if not copy.exists():
        copy.parent.mkdir(exist_ok=True)
        shutil.copy2(built, copy)
    return copy


def convert(work, name):
    """Convert the fresh SST case in `work` to the compared closure `name` in place."""
    from forge_fixed import add_library, frame_omega, render_turbulence_properties
    work = Path(work).resolve()
    meta = json.loads((work/'reproduce.json').read_text())
    if meta['closure'] != 'sst':
        sys.exit(f'{work} is a case of {meta["closure"]}, not of SST')
    times = [p for p in work.iterdir() if p.is_dir() and re.fullmatch(r'[0-9]+(\.[0-9]+)?', p.name) and float(p.name) > 0]
    if times or any((work/d).exists() for d in ('log.simpleFoam', 'processor0')):
        sys.exit(f'{work} has been run; convert a fresh SST case from make_case.py')
    turb = work/'constant/turbulenceProperties'
    if not re.search(r'(?m)^\s*RASModel\s+kOmegaSST\s*;', turb.read_text()):
        sys.exit(f'{turb} is not the stock SST dictionary')
    control = work/'system/controlDict'
    text = control.read_text()
    basis = re.findall(r'"([^"]*libforgeClosures-[0-9a-f]{12}\.so)"', text)
    if len(basis) != 1:
        sys.exit(f'{control} does not load the library of make_case.py exactly once')
    cases = {c['id']: c for c in json.loads((ROOT/'reproduce/cases.json').read_text())['cases']}
    model = comparator(name)
    control.write_text(add_library(text, basis[0], library(model)))
    turb.write_text(render_turbulence_properties(model, frame_omega(cases[meta['case_id']])))
    (work/'reproduce.json').write_text(json.dumps({**meta, 'closure': name}, indent=1) + '\n')
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', help='a fresh SST case written by make_case.py')
    ap.add_argument('comparator', choices=sorted(COMPARATORS))
    a = ap.parse_args()
    model = convert(a.case, a.comparator)
    print(f'{a.case}: SST case converted to {a.comparator} ({model.ras_model})')
    print(f'run:   {Path(a.case).resolve()}/Allrun')


if __name__ == '__main__':
    main()
