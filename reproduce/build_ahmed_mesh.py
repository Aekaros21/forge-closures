#!/usr/bin/env python3
"""Build the Ahmed-body mesh of the paper (slant 25 or 35 degrees; level coarse, base or fine).

    python reproduce/build_ahmed_mesh.py SLANT LEVEL [--np N]

Run in an OpenFOAM v2312 environment. The body and its stilts are written as STL surfaces by the
evaluation code (evaluation/tedp/holdout_cases/ahmed.py), then blockMesh, surfaceFeatureExtract and
snappyHexMesh (in parallel), reconstructParMesh and checkMesh build the mesh, which is copied to
data/assets/meshes/ahmed/s<SLANT>_<LEVEL>/polyMesh, where the ahmed_25 and ahmed_35 cases read it.
The paper used the base level. A report of the mesh size and quality is written next to the mesh.
snappyHexMesh in parallel is not bit-reproducible, so a rebuilt mesh can differ slightly from the
paper's; its cell count and quality should be close to those of Appendix B.
"""
import argparse, hashlib, json, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))
from tedp.holdout_cases import ahmed  # noqa: E402


def run(case, command, log):
    started = time.time()
    if subprocess.run(['bash', '-c', f'cd "{case}" && {command} > {log} 2>&1']).returncode:
        sys.exit(f'{command} failed; see {case/log}')
    return round(time.time() - started, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('slant', type=int, choices=(25, 35))
    ap.add_argument('level', choices=tuple(ahmed.LEVELS))
    ap.add_argument('--np', type=int, default=12, help="MPI ranks for snappyHexMesh (default 12, as for the paper's meshes)")
    a = ap.parse_args()
    tag = f's{a.slant}_{a.level}'
    case = ROOT/'runs/ahmed_mesh'/tag
    if case.exists():
        shutil.rmtree(case)
    ahmed.mesh_inputs(case, a.slant, a.level)
    (case/'system/controlDict').write_text(ahmed.control_dict(100))
    (case/'system/fvSchemes').write_text(ahmed.fv_schemes())
    (case/'system/fvSolution').write_text(ahmed.fv_solution())
    (case/'system/decomposeParDict').write_text(
        'FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n'
        f'numberOfSubdomains {a.np};\nmethod scotch;\n')
    timing = {'blockMesh': run(case, 'blockMesh', 'log.blockMesh'),
              'surfaceFeatureExtract': run(case, 'surfaceFeatureExtract', 'log.surfaceFeatureExtract'),
              'decomposePar': run(case, 'decomposePar -force', 'log.decomposePar'),
              'snappyHexMesh': run(case, f'mpirun -np {a.np} snappyHexMesh -parallel -overwrite', 'log.snappyHexMesh'),
              'reconstructParMesh': run(case, 'reconstructParMesh -constant', 'log.reconstructParMesh')}
    for p in case.glob('processor*'):
        shutil.rmtree(p)
    timing['checkMesh'] = run(case, 'checkMesh -constant', 'log.checkMesh')
    check = (case/'log.checkMesh').read_text()
    stats = {}
    for key, pattern in (('cells', r'cells:\s+(\d+)'), ('max_non_orthogonality', r'Mesh non-orthogonality Max:\s+([\d.eE+-]+)'),
                         ('max_skewness', r'Max skewness = ([\d.eE+-]+)')):
        m = re.search(pattern, check)
        stats[key] = float(m.group(1)) if m else None
    stats['mesh_ok'] = 'Mesh OK.' in check
    target = ROOT/'data/assets/meshes/ahmed'/tag/'polyMesh'
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(case/'constant/polyMesh', target)
    report = {'slant_deg': a.slant, 'level': a.level, 'level_settings': ahmed.LEVELS[a.level], 'mpi_ranks': a.np,
              'timing_s': timing, 'check_mesh': stats,
              'mesh_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()}}
    (target.parent/'mesh_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mesh': str(target.relative_to(ROOT)), **stats, 'timing_s': timing}))


if __name__ == '__main__':
    main()
