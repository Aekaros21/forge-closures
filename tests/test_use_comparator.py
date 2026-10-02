"""reproduce/use_comparator.py: an SST case becomes a case of a compared closure by two files, as in the campaign.

The case here is a stand-in with the three files the conversion reads (no mesh, no OpenFOAM).
"""
import importlib.util
import json
from pathlib import Path

import pytest

from forge_fixed import source_sha256
from tedp.casegen import render_turbulence_properties
from tedp.holdout_cases import rotchan

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('use_comparator', ROOT/'reproduce/use_comparator.py')
uc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uc)
BASIS = '/x/build/libforgeClosures-0123456789ab.so'
CONTROL = ('FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication simpleFoam;\n'
           'functions\n{\n    probes\n    {\n        libs            (fieldFunctionObjects);\n    }\n}\n'
           f'\nlibs            ("{BASIS}");\n')


def sst_case(root, case_id='rotchan_ro10', name='rotchan_ro10_sstrc'):
    work = root/name
    (work/'constant').mkdir(parents=True)
    (work/'system').mkdir()
    (work/'0').mkdir()
    (work/'constant/turbulenceProperties').write_text(render_turbulence_properties(None, mode='expressions'))
    (work/'system/controlDict').write_text(CONTROL)
    (work/'reproduce.json').write_text(json.dumps({'case_id': case_id, 'closure': 'sst', 'nprocs': 1, 'paper_nprocs': 1}))
    return work


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(uc, 'library', lambda model: tmp_path/f'{Path(model.library).stem}-abcdefabcdef.so')
    return tmp_path


def test_the_comparator_sources_are_those_of_the_paper():
    models = json.loads((ROOT/'closures/comparators/comparator_models.json').read_text())['models']
    fixed = [m for m in models if m.get('fixed')]
    assert sorted(m['label'] for m in fixed) == sorted(uc.COMPARATORS.values())
    for m in fixed:
        assert source_sha256(ROOT/'closures/comparators'/Path(m['fixed']['source_dir']).name) == m['fixed']['source_sha256']


@pytest.mark.parametrize('name,ras', [('sstrc', 'kOmegaSSTRC'), ('earsm', 'kOmegaEARSM'), ('autoturb', 'kOmegaSSTAutoTurb')])
def test_conversion_changes_the_model_the_library_and_nothing_else(built, name, ras):
    work = sst_case(built, name=f'rotchan_ro10_{name}')
    uc.convert(work, name)
    turb = (work/'constant/turbulenceProperties').read_text()
    assert f'RASModel        {ras};' in turb and f'{ras}Coeffs' in turb
    assert f'frameOmega      (0 0 {rotchan.omega_for(0.1)});' in turb
    control = (work/'system/controlDict').read_text()
    assert control == CONTROL.replace(f'"{BASIS}"', f'"{BASIS}" "{built}/lib{ras}-abcdefabcdef.so"')
    assert json.loads((work/'reproduce.json').read_text())['closure'] == name


def test_a_non_rotating_case_gets_no_frame_rotation(built):
    work = sst_case(built, 'tmr_plate_137x97', 'plate_sstrc')
    uc.convert(work, 'sstrc')
    assert 'frameOmega      (0 0 0);' in (work/'constant/turbulenceProperties').read_text()


def test_only_a_fresh_sst_case_is_converted(built):
    work = sst_case(built)
    uc.convert(work, 'sstrc')
    with pytest.raises(SystemExit, match='not of SST'):
        uc.convert(work, 'earsm')
    ran = sst_case(built, name='ran')
    (ran/'100').mkdir()
    with pytest.raises(SystemExit, match='has been run'):
        uc.convert(ran, 'sstrc')
