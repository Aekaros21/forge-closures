#!/usr/bin/env python3
"""Download the reference data that this repository cannot redistribute, and check all inputs.

    python data/fetch_data.py pch22    rotating-channel DNS (AGARD PCH22, Kristoffersen & Andersson 1993),
                                       public, from the AGARD database mirror at Universidad Politecnica de Madrid
    python data/fetch_data.py ducts    square-duct DNS and SST template cases (McConkey et al. 2021, Kaggle),
                                       through the Kaggle command-line client (pip install kaggle; an API token in
                                       ~/.kaggle/kaggle.json); only the files the cases need are downloaded
    python data/fetch_data.py check    list every input of the reproducible cases, with its checksum status

The NASA TMR grids and reference data, the NASA hump data and the FAITH hill data are public domain and are
included in this repository. The ERCOFTAC data of the Ahmed bodies, the Stanford diffuser and the wing-body
junction require registration with ERCOFTAC; see data/README.md for the files and where they go.
Every downloaded file is checked against data/SHA256SUMS, the checksums of the files used in the paper;
'check' also verifies ERCOFTAC files placed by hand.
"""
import hashlib, ssl, subprocess, sys, tempfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = ROOT/'data/SHA256SUMS'
PCH22 = 'https://torroja.dmt.upm.es/turbdata/agard/chapter5/PCH22/f2/'
ROTATION = ('00', '01', '05', '10', '15', '20', '50')
KAGGLE_DATASET = 'ryleymcconkey/ml-turbulence-dataset'
DUCTS = (2000, 3200)          # the development and holdout ducts of reproduce/cases.json
MESH = ('boundary', 'cellZones', 'faceZones', 'faces', 'neighbour', 'owner', 'pointZones', 'points')
FIELDS = ('U', 'epsilon', 'f', 'k', 'nut', 'omega', 'p', 'phit')
SYSTEM = ('blockMeshDict', 'controlDict', 'decomposeParDict', 'fvOptions', 'fvSchemes', 'fvSolution')
ERCOFTAC = tuple(f'data/assets/references/{d}/' for d in ('ahmed', 'diffuser3d', 'wingbody'))


def sums():
    return {line.split(None, 1)[1].strip(): line.split()[0] for line in SUMS.read_text().splitlines() if line.strip()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path):
    rel = str(Path(path).relative_to(ROOT))
    want = sums().get(rel)
    if want and sha(path) != want:
        raise SystemExit(f'checksum mismatch: {rel}')
    return want is not None


def fetch_pch22():
    # The mirror serves an incomplete certificate chain; integrity rests on the SHA-256 checksums below.
    context = ssl.create_default_context()
    context.check_hostname, context.verify_mode = False, ssl.CERT_NONE
    for ro in ROTATION:
        name = f'f2_r_{ro}.dat'
        dest = ROOT/'data/assets/references/rotation/f2'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PCH22 + name, context=context, timeout=60) as r:
            dest.write_bytes(r.read())
        verify(dest)
        print('ok', dest.relative_to(ROOT))


def duct_plan(re_b):
    case = f'squareDuct_Re_{re_b}'
    dns_src, dns_dst = f'foam/DNS/squareDuct/{case}', ROOT/'data/assets/references/duct'/case
    rans_src, rans_dst = f'foam/komegasst/squareDuct/{case}', ROOT/'data/mcconkey/foam/komegasst/holdout_duct'/case
    items = [(f'{dns_src}/0/{f}', dns_dst/'0'/f) for f in ('U', 'tau')]
    items += [(f'{dns_src}/constant/polyMesh/{m}', dns_dst/'constant/polyMesh'/m) for m in MESH]
    items += [(f'{dns_src}/constant/transportProperties', dns_dst/'constant/transportProperties')]
    items += [(f'{rans_src}/0.orig/{f}', rans_dst/'0'/f) for f in FIELDS]
    items += [(f'{rans_src}/constant/polyMesh/{m}', rans_dst/'constant/polyMesh'/m) for m in MESH]
    items += [(f'{rans_src}/constant/{n}', rans_dst/'constant'/n) for n in ('transportProperties', 'turbulenceProperties')]
    items += [(f'{rans_src}/system/{n}', rans_dst/'system'/n) for n in SYSTEM]
    return case, dns_dst, rans_dst, items


def fetch_ducts():
    for re_b in DUCTS:
        case, dns, rans, items = duct_plan(re_b)
        for remote, dest in items:
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as tmp:
                    subprocess.run(['kaggle', 'datasets', 'download', KAGGLE_DATASET, '-f', remote, '-p', tmp], check=True)
                    got = Path(tmp)/Path(remote).name
                    if not got.is_file():            # the client zips large files
                        subprocess.run(['unzip', '-o', '-q', str(got) + '.zip', '-d', tmp], check=True)
                    got.replace(dest)
            verify(dest)
        # the scorer's reference arrays are rebuilt from the DNS case, as for the paper
        sys.path.insert(0, str(ROOT/'evaluation'))
        from tedp.holdout_cases import duct_scoring as ds
        ds.save_reference(ds.build_reference(dns, rans), ROOT/f'data/mcconkey/cache/holdout_duct_{case}.npz')
        print('ok', case)


def check():
    groups = {  # source: (path prefixes, how to obtain the files)
        'included in this repository': (('data/assets/grids/', 'data/assets/references/tmr/',
                                         'data/assets/references/hump/', 'cases/faith/'), 'restore them with git'),
        'rotating-channel DNS': (('data/assets/references/rotation/',), 'python3 data/fetch_data.py pch22'),
        'square-duct DNS and templates': (('data/assets/references/duct/', 'data/mcconkey/'),
                                          'python3 data/fetch_data.py ducts'),
        'ERCOFTAC (Ahmed body, diffuser, wing-body junction)': (ERCOFTAC, 'download by hand, see data/README.md'),
    }
    status = {name: [] for name in groups}
    for rel, want in sorted(sums().items()):
        name = next(n for n, (prefixes, _) in groups.items() if rel.startswith(prefixes))
        p = ROOT/rel
        status[name].append((rel, 'missing' if not p.is_file() else ('ok' if sha(p) == want else 'DIFFERENT')))
    for name, files in status.items():
        bad = [(rel, s) for rel, s in files if s != 'ok']
        if not bad:
            print(f'{name}: all {len(files)} files present and verified')
            continue
        missing = sum(s == 'missing' for _, s in bad)
        print(f'{name}: {missing} of {len(files)} files missing' + (f' ({groups[name][1]})' if missing else '')
              + (f', {len(bad) - missing} DIFFERENT from the paper' if len(bad) > missing else ''))
        for rel, s in bad:
            if s != 'missing' or missing < len(files):
                print(f'    {s:9s} {rel}')


if __name__ == '__main__':
    commands = {'pch22': fetch_pch22, 'ducts': fetch_ducts, 'check': check}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
