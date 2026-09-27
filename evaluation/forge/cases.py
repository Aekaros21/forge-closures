"""Independent v2 case adapters. Preparation never runs a solver.

Scoring reads an explicit written snapshot and refuses missing required fields;
the execution layer owns postprocessing and numerical qualification.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import threading

import numpy as np
from fluidfoam import readmesh, readscalar, readvector, readtensor, readsymmtensor

from tedp import foammesh, tier2
from tedp.casegen import render_turbulence_properties
from tedp.data import mcconkey
from tedp.spec import to_json
from tedp.holdout_cases import duct_scoring as ds, holdout_scoring as hs
from tedp.holdout_cases import hump, rotchan, tmr_flatplate, tmr_bump, tmr_naca, faith, rectduct, ahmed, diffuser3d, wingbody, lmchannel

ROOT = Path(__file__).resolve().parents[2]
_GENERATOR_LOCK = threading.RLock()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _set_top_level(text,key,value):
    """Change a root dictionary entry, preserving function-object controls."""
    masked=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                  lambda m: ''.join('\n' if c=='\n' else ' ' for c in m[0]),text)
    matches=[m for m in re.finditer(r'(?m)^(\s*)'+re.escape(key)+r'\s+[^;\n]+;',text)
             if masked[:m.start()].count('{')==masked[:m.start()].count('}')]
    if len(matches)>1:raise ValueError(f'Ambiguous top-level dictionary key: {key}')
    if matches:
        m=matches[0];return text[:m.start()]+f'{m[1]}{key} {value};'+text[m.end():]
    return text+f'\n{key} {value};\n'


def _work_path(work, *, fresh=False):
    work = Path(work).absolute()
    if not any(work.resolve().is_relative_to((ROOT / d).resolve()) and work.resolve() != (ROOT / d).resolve()
               for d in ('runs', 'reports')):
        raise ValueError('Case output must be below this v2 repository runs/ or reports/')
    if fresh and work.exists():
        raise FileExistsError(f'Refusing to replace an existing case: {work}')
    return work


@contextmanager
def _grid_root(module, directory):
    with _GENERATOR_LOCK:
        old = module.GRID_ROOT
        module.GRID_ROOT = directory
        try:
            yield
        finally:
            module.GRID_ROOT = old


def prepare_case(case, spec, work, *, end_time, library):
    work = _work_path(work, fresh=True)
    library = Path(library).resolve()
    if not library.is_file() or not library.is_relative_to(ROOT.resolve()):
        raise ValueError('An existing independent v2 library is required')
    digest = _sha(library)
    if digest[:12] not in library.name:
        raise ValueError('Use an immutable hash-named library, not a mutable build output')
    work.parent.mkdir(parents=True, exist_ok=True)
    adapter, conf = case['adapter'], case['adapter_config']
    endpoint = int(end_time)
    if endpoint <= 0:
        raise ValueError('end_time must be positive')
    commands, info = [], {}
    if adapter in ('mcconkey', 'duct'):
        source = mcconkey.case_dir(conf['case_id'])
        if not source.resolve().is_relative_to((ROOT/'data').resolve()):
            raise ValueError('Case templates must be independent v2 assets')
        work.mkdir()
        for folder in ('0', 'constant', 'system'):
            shutil.copytree(source/folder, work/folder, symlinks=False)
        (work/'constant/turbulenceProperties').write_text(render_turbulence_properties(spec, mode='expressions'))
        info = {'template': str(source), 'initialization': 'cold_shipped_initial_fields'}
        if adapter == 'duct' and conf.get('streamwise_cells') is not None:
            count = int(conf['streamwise_cells'])
            if count < 1 or count > 75:
                raise ValueError('Duct streamwise cell count must be in [1,75]')
            # Fully developed cyclic geometry: preserve the complete 96x96
            # cross-section and solve all three velocity components. This is
            # a separately identified mesh, requiring axial-resolution checks.
            for field in (work/'0').iterdir():
                if field.is_file() and 'nonuniform' in field.read_text(errors='replace'):
                    raise ValueError('Reduced duct requires uniform initial fields or explicit conservative mapping')
            block = work/'system/blockMeshDict'
            text, changes = re.subn(r'(?m)^\s*nx\s+75\s*;', f'nx {count};', block.read_text())
            if changes != 1:
                raise ValueError('Unexpected duct block topology')
            half = int(conf.get('cross_section_cells_per_half', 48))
            if half != 48:
                # Verification mesh study: refine every quadrant block, same wall grading ratio.
                if not 24 <= half <= 192:
                    raise ValueError('Duct cross-section cells per half must be in [24,192]')
                text, a = re.subn(r'(?m)^(\s*nyz\s+)48(\s*;)', rf'\g<1>{half}\g<2>', text)
                text, b = re.subn(r'(?m)^(\s*nyzp\s+)48(\s*;)', rf'\g<1>{half}\g<2>', text)
                if (a, b) != (1, 1):
                    raise ValueError('Unexpected duct cross-section definition')
            block.write_text(text)
            shutil.rmtree(work/'constant/polyMesh')
            commands = [['blockMesh']]
            info.update(streamwise_cells=count, cross_section_cells=[2*half,2*half],
                        assumption='fully developed, streamwise-periodic mean flow; all velocity components retained')
    elif adapter == 'rotation':
        ny = int(conf.get('cells', rotchan.NY))
        if not 32 <= ny <= 2048:
            raise ValueError('Rotating-channel wall-normal cell count must be in [32,2048]')
        grading = float(conf.get('wall_grading', rotchan.WALL_GRADING))
        re_half = float(conf.get('re_bulk_half', rotchan.RE_BULK_HALF))
        rotchan.make_case(work, float(conf['ro']), spec, end_time=endpoint, frame_rotation=True, ny=ny,
                          wall_grading=grading, re_bulk_half=re_half)
        commands = [['blockMesh']]
        info = {'ro': conf['ro'], 'cells': ny, 'frame_rotation': True, 'wall_grading': grading,
                're_bulk_half': re_half}
    elif adapter in ('plate', 'bump', 'naca'):
        module = {'plate': tmr_flatplate, 'bump': tmr_bump, 'naca': tmr_naca}[adapter]
        kwargs = {'alpha_deg': float(conf.get('alpha_deg', 0))} if adapter == 'naca' else {}
        with _grid_root(module, ROOT/'data/assets/grids'/adapter):
            info = module.make_case(work, conf['grid'], spec, end_time=endpoint, **kwargs)
    elif adapter == 'hump':
        info = hump.make_case(work, ROOT/'data/assets/grids/hump'/hump.GRIDS[conf['grid']], spec, end_time=endpoint)
    elif adapter == 'rectduct':
        # Verification-only 3D case: the development square-duct template (initial fields,
        # numerics, meanVelocityForce drive U_b = 0.482) with a rectangular cross-section and the
        # viscosity of the DNS bulk Reynolds number; see tedp.holdout_cases.rectduct.
        source = mcconkey.case_dir(conf.get('template_case', 'squareDuct_Re_2000'))
        if not source.resolve().is_relative_to((ROOT/'data').resolve()):
            raise ValueError('Case templates must be independent v2 assets')
        work.mkdir()
        for folder in ('0', 'constant', 'system'):
            shutil.copytree(source/folder, work/folder, symlinks=False)
        for field in (work/'0').iterdir():
            if field.is_file() and 'nonuniform' in field.read_text(errors='replace'):
                raise ValueError('Rectangular duct requires uniform template initial fields')
        drive = (work/'system/fvOptions').read_text()
        if not re.search(r'Ubar\s*\(\s*0\.482\s+0\s+0\s*\)', drive):
            raise ValueError('Unexpected duct drive; the reference scaling assumes U_b = 0.482')
        (work/'constant/turbulenceProperties').write_text(render_turbulence_properties(spec, mode='expressions'))
        shutil.rmtree(work/'constant/polyMesh')
        ratio = float(conf['aspect_ratio'])
        (work/'system/blockMeshDict').write_text(rectduct.block_mesh_dict(ratio))
        nu = rectduct.U_BULK * rectduct.H / float(conf['re_b'])
        props = work/'constant/transportProperties'
        text, changes = re.subn(r'(?m)^(\s*nu\s+(?:\[[^\]]*\]\s*)?)[0-9.eE+-]+(\s*;)',
                                lambda m: f'{m.group(1)}{nu:.10g}{m.group(2)}', props.read_text())
        if changes != 1:
            raise ValueError('Unexpected duct transportProperties')
        props.write_text(text)
        commands = [['blockMesh']]
        info = {'template': str(source), 'aspect_ratio': ratio, 're_b': float(conf['re_b']), 'nu': nu,
                'cells': rectduct.cells(ratio), 'initialization': 'cold_shipped_initial_fields',
                'assumption': 'fully developed, streamwise-periodic mean flow; all velocity components retained'}
    elif adapter == 'faith':
        # Verification-only 3D case. The block mesh is generated on the compute node and then
        # moved onto the analytic hill surface by post_mesh(); see tedp.holdout_cases.faith.
        mesh = conf.get('mesh', {})
        info = faith.write_case(work, spec, nx=int(mesh.get('nx', 260)), ny=int(mesh.get('ny', 90)),
                                nz=int(mesh.get('nz', 80)), end_time=endpoint,
                                y_grading=float(mesh.get('y_grading', 4000.0)),
                                x_grading=float(mesh.get('x_grading', 1.0)))
        commands = [['blockMesh'], ['post_mesh:faith_displace_onto_hill']]
    elif adapter == 'diffuser3d':
        # Verification-only 3D diffuser: unit block from blockMesh, mapped onto the walls by
        # post_mesh() on the compute node; see tedp.holdout_cases.diffuser3d.
        work.mkdir()
        info = diffuser3d.write_case(work, spec, level=conf.get('mesh_level', 'base'), end_time=endpoint)
        commands = [['blockMesh'], ['post_mesh:diffuser3d_map_onto_walls']]
    elif adapter == 'wingbody':
        # Verification-only 3D wing-body junction: the structured mesh and the measured inflow
        # boundary layer are written by Python (deterministic); see tedp.holdout_cases.wingbody.
        work.mkdir()
        info = wingbody.write_case(work, spec, level=conf.get('mesh_level', 'base'), end_time=endpoint)
        commands = []
    elif adapter == 'ahmed':
        # Verification-only 3D bluff body on a frozen snappyHexMesh mesh (a hashed asset built
        # once by scripts/verify_build_ahmed_mesh.py); see tedp.holdout_cases.ahmed.
        work.mkdir()
        info = ahmed.write_case(work, spec, int(conf['slant_deg']), ROOT/conf['mesh_dir'], endpoint)
        commands = []
    else:
        raise ValueError(f'Unsupported adapter: {adapter}')
    control = work/'system/controlDict'
    text = control.read_text()
    for key, value in [('startFrom', 'startTime'), ('startTime', '0'), ('endTime', str(endpoint)),
                       ('writeInterval', str(max(1, endpoint//4))), ('purgeWrite', '0'), ('writePrecision', '12')]:
        text = _set_top_level(text,key,value)
    text = text.replace('"libkOmegaSSTBasis.so"', '"'+str(library)+'"')
    if str(library) not in text:
        text += '\nlibs ("'+str(library)+'");\n'
    control.write_text(text)
    turb = work/'constant/turbulenceProperties'
    if spec is not None:
        text = turb.read_text()
        text = re.sub(r'(kOmegaSSTBasisCoeffs\s*\{)', r'\1\n        coupledStressBoundary true;', text, count=1)
        turb.write_text(text)
    solution = work/'system/fvSolution'
    relaxation = case['protocol'].get('omega_relaxation')
    if relaxation is not None:
        if not 0 < float(relaxation) <= 1:
            raise ValueError('Invalid omega relaxation')
        text, count = re.subn(r'(relaxationFactors\s*\{\s*equations\s*\{)',
            lambda m: m[0]+f'\n        omega {float(relaxation)};', solution.read_text(), count=1)
        if count != 1:
            raise ValueError('Missing omega equation relaxation block')
        solution.write_text(text)
    nprocs = int(case['protocol'].get('nprocs', 1))
    (work/'system/decomposeParDict').write_text(
        'FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n'
        f'numberOfSubdomains {nprocs};\nmethod scotch;\n')
    metadata = {'case_id': case['id'], 'family': case['family'], 'group': case['group'],
        'adapter': adapter, 'case_path': str(work), 'end_time': endpoint, 'nprocs': nprocs,
        'spec': json.loads(to_json(spec)) if spec is not None else None,
        'library': str(library), 'library_sha256': digest, 'pre_commands': commands,
        'required_postprocess': ['grad(U)', 'wallShearStress'], 'generator_info': info,
        'initial_fields_sha256': {p.name: _sha(p) for p in (work/'0').iterdir() if p.is_file()},
        'numerics_sha256': {name: _sha(work/'system'/name) for name in
                           ('controlDict', 'fvSchemes', 'fvSolution', 'decomposeParDict')},
        'turbulence_sha256': _sha(turb), 'reference': case['reference']}
    (work/'forge_case.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def post_mesh(case, work):
    """Adapter geometry step that must follow mesh generation; None for adapters without one.

    Runs on the compute node right after blockMesh, before decomposition. Only the FAITH hill
    uses it: its Cartesian block is moved onto the analytic hill surface (terrain following).
    """
    if case['adapter'] == 'diffuser3d':
        return diffuser3d.map_onto_walls(_work_path(work))
    if case['adapter'] != 'faith':
        return None
    n_points, peak = faith.displace_onto_hill(_work_path(work))
    # ``peak`` is the analytic hill height at the highest mesh vertex. It equals H only when a
    # vertex falls on the crest (the base mesh); the coarse and fine meshes straddle it (0.18% and
    # 0.06% below H). Require the crest to be resolved to within 1% and never exceeded.
    if not faith.H*0.99 <= peak <= faith.H*(1+1e-9):
        raise ValueError(f'FAITH hill peak {peak} m does not resolve the analytic {faith.H} m crest')
    return {'displaced_points': n_points, 'hill_peak_m': peak, 'crest_deficit': 1.0-peak/faith.H}


def _time(work, time):
    if time is not None:
        t = str(time)
        if not re.fullmatch(r'\d+(?:\.\d+)?',t):
            raise ValueError('Snapshot time must be a nonnegative numeric directory')
    else:
        times = [p.name for p in work.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?', p.name)]
        if not times:
            raise FileNotFoundError('No written fields')
        t = max(times, key=float)
    if not (work/t/'U').is_file():
        raise FileNotFoundError(work/t/'U')
    return t


def _fields(work, t):
    u = np.asarray(readvector(str(work), t, 'U', verbose=False)).T
    k = np.asarray(readscalar(str(work), t, 'k', verbose=False))
    nut = np.asarray(readscalar(str(work), t, 'nut', verbose=False))
    path = work/t/'grad(U)'
    if not path.exists():
        raise FileNotFoundError(f'Execution layer must write grad(U): {path}')
    g = np.asarray(readtensor(str(work), t, 'grad(U)', verbose=False))
    uv = -nut*(g[1]+g[3])
    if (work/t/'nonlinearStress').exists():
        uv += np.asarray(readsymmtensor(str(work), t, 'nonlinearStress', verbose=False))[1]
    if not all(np.isfinite(a).all() for a in (u,k,nut,g,uv)):
        raise ValueError('Nonfinite fields cannot be scored')
    return u,k,nut,uv


def _score_values(e_u, e_cf, e_uv):
    return {'velocity_rmse': float(e_u), 'skin_friction_rmse': float(e_cf),
            'shear_stress_rmse': float(e_uv), 'composite': float(.5*e_u+.3*e_cf+.2*e_uv)}


def _duct_snapshot(work,t,ref):
    u,k,nut,uv=_fields(work,t)
    grad=np.asarray(readtensor(str(work),t,'grad(U)',verbose=False))
    uxz=-nut*(grad[2]+grad[6])
    if (work/t/'nonlinearStress').exists():
        uxz+=np.asarray(readsymmtensor(str(work),t,'nonlinearStress',verbose=False))[2]
    x,y,z=ds.foammesh_cell_centres(work)
    field=np.column_stack([u,uv,uxz]);points,values=ds._column_average(y,z,field)
    walls,labels,_=ds.wall_faces(work)
    wp,_=ds._column_average(walls[:,1],walls[:,2],np.zeros((len(walls),5)))
    sample=ds._interpolate(np.vstack([points,wp]),np.vstack([values,np.zeros((len(wp),5))]),ref['points'])
    du=sample[:,:3]-ref['u'];shear_delta=sample[:,3:5]-np.column_stack([ref['tau_xy'],ref['tau_xz']])
    wc,labels,tau=ds.wall_shear(work,u,ref['nu']);cf=tau/(.5*ref['u_bulk']**2);sampled_cf=np.full(len(ref['cf']),np.nan)
    for wall in ds.WALLS:
        target=ref['wall_labels']==wall;source=labels==wall
        if not target.any():continue
        if not source.any():raise ValueError(f'Missing duct wall {wall}')
        free=2 if wall.startswith('y') else 1
        p,v=ds._column_average(wc[source,free],np.zeros(source.sum()),cf[source]);order=np.argsort(p[:,0])
        sampled_cf[target]=np.interp(ref['wall_points'][target,free],p[order,0],v[order,0])
    errors=_score_values(np.sqrt(np.mean(np.sum(du**2,axis=1)))/ref['u_bulk'],
        np.sqrt(np.mean((sampled_cf-ref['cf'])**2)),np.sqrt(np.mean(np.sum(shear_delta**2,axis=1)))/ref['u_bulk']**2)
    a,b=sample[:,1:3],ref['u'][:,1:3];ra=float(np.sqrt(np.mean(np.sum(a*a,axis=1))));rb=float(np.sqrt(np.mean(np.sum(b*b,axis=1))))
    denom=float(np.linalg.norm(a)*np.linalg.norm(b));corr=float(np.sum(a*b)/denom) if denom>0 else 0.
    errors['secondary_velocity_rmse']=float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1)))/ref['u_bulk'])
    return errors,{'secondary_magnitude_ratio':ra/rb,'secondary_correlation':corr,
        'secondary_dns_projection_ratio':float(np.sum(a*b)/max(float(np.sum(b*b)),1e-30)),
        'secondary_rms_model_over_bulk':ra/ref['u_bulk'],'secondary_rms_dns_over_bulk':rb/ref['u_bulk'],
        'secondary_orientation_resolved':ra>1e-8*rb,'wall_cf':sampled_cf.tolist(),
        'secondary_projection':'Streamwise column average with no-slip boundary points, interpolated to original DNS coordinates'}


def _numeric_rows(path, columns=2, *, first_zone=False):
    rows=[]; started=False
    for line in Path(path).read_text().splitlines():
        if line.strip().lower().startswith('zone'):
            if first_zone and started and rows:break
            started=True
        try: values=[float(x.replace('D','E')) for x in line.replace(',',' ').split()]
        except ValueError: continue
        if len(values)>=columns: rows.append(values[:columns])
    if not rows:raise ValueError(f'No numeric reference rows: {path}')
    return np.asarray(rows)


def _wall(work,t,patch):
    path=work/t/'wallShearStress'
    if not path.exists():raise FileNotFoundError(f'Execution layer must write wallShearStress: {path}')
    geom=foammesh.wall_geometry(work,patch)
    tau=np.asarray(readvector(str(work),t,'wallShearStress',boundary=patch,verbose=False))
    if tau.ndim==1:tau=np.repeat(tau[:,None],len(geom.owner_cells),axis=1)
    p=np.atleast_1d(np.asarray(readscalar(str(work),t,'p',verbose=False)))
    pressure=p[geom.owner_cells] if len(p)>1 else np.full(len(geom.owner_cells),p[0])
    return geom,tau.T,pressure


def surface_force_coefficients(face_areas,tau,p,span,sref,alpha_deg):
    """Independent wall integration, Uref=rho=1, OpenFOAM outward fluid normals."""
    face_areas=np.asarray(face_areas);area=np.linalg.norm(face_areas,axis=1)
    if span<=0 or sref<=0 or (area<=0).any():raise ValueError('Invalid force normalization/geometry')
    force=2*(np.sum(np.asarray(p)[:,None]*face_areas,axis=0)-np.sum(np.asarray(tau)*area[:,None],axis=0))/(span*sref)
    a=math.radians(alpha_deg)
    return {'drag_coefficient':float(force@np.array([math.cos(a),math.sin(a),0.])),
            'lift_coefficient':float(force@np.array([-math.sin(a),math.cos(a),0.]))}


def force_history_comparison(work,t,alpha_deg,coefficients):
    """Compare wall integration to the actual forceCoeffs row at this snapshot.

    Missing history never produces a passed check. No interpolation across
    snapshots/restarts, because that could conceal an endpoint mismatch.
    """
    found=None
    for path in sorted(work.glob('postProcessing/forceHistory/*/coefficient*.dat')):
        for line in path.read_text().splitlines():
            if line.lstrip().startswith('#'):continue
            try:values=[float(x) for x in line.split()]
            except ValueError:continue
            if len(values)>=5 and values[0]==float(t):found=(path,values)
    if found is None:return {'available':False,'passed':False,'reason':'No forceCoeffs record at the scored snapshot'}
    path,v=found;a=math.radians(alpha_deg)
    expected={'drag_coefficient':v[1]*math.cos(a)+v[4]*math.sin(a),
              'lift_coefficient':-v[1]*math.sin(a)+v[4]*math.cos(a)}
    delta={name:coefficients[name]-val for name,val in expected.items()}
    return {'available':True,'source':str(path),'absolute_differences':delta,
            'force_coeffs':expected,'passed':all(abs(delta[k])<=1e-8+1e-4*abs(expected[k]) for k in expected),
            'tolerance':'absolute1e-8 + relative1e-4; extraction identity check, not physics accuracy'}


_CENTRES = {}


def _cell_centres(work):
    """Cell centres of a (static) mesh, read once per case directory."""
    key = str(work)
    if key not in _CENTRES:
        _CENTRES.clear()
        _CENTRES[key] = np.asarray(readmesh(key, verbose=False)).T
    return _CENTRES[key]


def score_case(case, work, *, time=None):
    work=_work_path(work);t=_time(work,time);adapter=case['adapter'];conf=case['adapter_config']
    errors={};obs={'time':t};reference=dict(case['reference'])
    if adapter in ('duct','rectduct'):
        ref=ds.load_reference(ROOT/conf['reference_npz'])
        errors,diag=_duct_snapshot(work,t,ref);obs.update(diag)
    elif adapter=='mcconkey':
        u,k,nut,uv=_fields(work,t);data=tier2.case_data(conf['case_id']);ref=data.ref
        cf=tier2._wall_shear(work,u,data.nu)/(.5*ref.u_bulk**2)
        du=u-ref.u_ref
        errors=_score_values(np.sqrt(np.mean(np.sum(du**2,axis=1)))/ref.u_bulk,
            np.sqrt(np.mean((cf-ref.cf_ref)**2)), np.sqrt(np.mean((uv-ref.uv_ref)**2))/ref.u_bulk**2)
        table=mcconkey.load_case_table(conf['case_id'],'REF')
        errors['turbulent_k_rmse']=float(np.sqrt(np.mean((k-table['REF_k'])**2))/ref.u_bulk**2)
        obs.update(wall_cf=cf.tolist(),velocity_mse_streamwise=float(np.mean(du[:,0]**2)),
                   velocity_mse_vector=float(np.mean(np.sum(du**2,axis=1))))
    elif adapter=='rotation':
        u,k,nut,uv=_fields(work,t);xyz=np.asarray(readmesh(str(work),verbose=False)).T;idx=np.argsort(xyz[:,1]);y=xyz[idx,1]
        cf=[]
        for patch in ('bottomWall','topWall'):
            _,tau,_=_wall(work,t,patch);cf.append(float(2*np.mean(abs(tau[:,0]))))
        ref=(lmchannel.reference() if conf.get('reference_kind')=='lee_moser'
             else hs.pch22_reference(ROOT/conf['reference_file']))
        score=hs.score_rotchan(y,u[idx,0],-uv[idx],tuple(cf),np.column_stack([ref['y'],ref['u']]),
                              np.column_stack([ref['y'],ref['muv']]),tuple(ref['cf']))
        errors=_score_values(score.e_u,score.e_cf,score.e_uv)
        ratio=cf[0]/cf[1];rr=float(ref['cf'][0]/ref['cf'][1])
        errors['wall_friction_asymmetry_error']=abs(ratio-rr)
        obs.update(wall_cf=cf,wall_cf_reference=list(ref['cf']),wall_friction_ratio=ratio,
                   wall_cf_bottom=cf[0],wall_cf_top=cf[1],
                   wall_friction_reference_ratio=rr,y=y.tolist(),u=u[idx,0].tolist(),
                   uv=uv[idx].tolist(),reference_uv_is_budget_derived=True)
    elif adapter=='hump':
        u,k,nut,uv=_fields(work,t);xyz=np.asarray(readmesh(str(work),verbose=False)).T
        geom,tau,_=_wall(work,t,'hump');idx=np.argsort(geom.face_centres[:,0]);x=geom.face_centres[idx,0];cf=-2*tau[idx,0]
        base=ROOT/'data/assets/references/hump';xr,cr=hs.read_hump_wall(base/'noflow_cf.exp.dat','cf');piv=hs.read_hump_piv(base/'noflow_vel_and_turb.exp.dat')
        score=hs.score_hump(xyz[:,:2],u,uv,x,cf,piv,xr,cr)
        errors=_score_values(score.e_u,score.e_cf,score.e_uv)
        crosses=[]
        for i in range(len(x)-1):
            if cf[i]*cf[i+1]<0 and .3<x[i]<2:
                crosses.append({'x':float(x[i]-cf[i]*(x[i+1]-x[i])/(cf[i+1]-cf[i])),
                                'kind':'separation' if cf[i]>0 else 'reattachment'})
        obs.update(wall_x=x.tolist(),wall_cf=cf.tolist(),crossings=crosses)
    elif adapter=='faith':
        velocity=faith.score_centreline(work,t)
        geom,tau,_=_wall(work,t,'bottom');centres=geom.face_centres
        # the row of wall faces next to the symmetry plane is the measured centreline
        row=np.isclose(centres[:,2],centres[:,2].min(),rtol=0.,atol=1e-6*faith.H)
        idx=np.argsort(centres[row,0]);x_h=centres[row,0][idx]/faith.H
        cf=(-2.*tau[row,0]/faith.U_REF**2)[idx]
        crossings=faith.centreline_crossings(x_h,cf);bubble=faith.bubble_errors(crossings)
        errors={'velocity_rmse':velocity['composite_velocity'],
                'separation_position_error':bubble['separation_position_error'],
                'reattachment_position_error':bubble['reattachment_position_error']}
        obs.update(wall_x=x_h.tolist(),wall_cf=cf.tolist(),crossings=crossings,
                   rmse_u=velocity['rmse_u'],rmse_v=velocity['rmse_v'],piv_points=velocity['n_u'],
                   separation_x_over_h=bubble['separation_x_over_h'],
                   reattachment_x_over_h=bubble['reattachment_x_over_h'])
        reference.update(measured_separation_x_over_h=faith.MEASURED_SEPARATION_XH,
                         measured_reattachment_x_over_h=faith.MEASURED_REATTACHMENT_XH)
    elif adapter=='diffuser3d':
        centres=_cell_centres(work)
        u=np.asarray(readvector(str(work),t,'U',verbose=False)).T
        geom,tau,wall_p=_wall(work,t,'walls')
        errors,diag=diffuser3d.score(centres,u,geom.face_centres,wall_p)
        obs.update(diag)
        reference.update(source='Cherry, Elkins and Eaton 2008, ERCOFTAC UFR 4-16 (Diffuser 1)',u_bulk=diffuser3d.U_BULK)
    elif adapter=='wingbody':
        centres=_cell_centres(work)
        u=np.asarray(readvector(str(work),t,'U',verbose=False)).T
        p=np.asarray(readscalar(str(work),t,'p',verbose=False))
        floor,_,floor_p=_wall(work,t,'floor')
        wing,_,wing_p=_wall(work,t,'wing')
        errors,diag=wingbody.score(centres,u,p,(floor.face_centres,floor_p),(wing.face_centres,wing_p))
        obs.update(diag)
        reference.update(source='Devenport and Simpson 1990; Fleming et al. 1993 (ERCOFTAC case 008)',u_ref=wingbody.U_REF)
    elif adapter=='ahmed':
        centres=_cell_centres(work)
        u=np.asarray(readvector(str(work),t,'U',verbose=False)).T
        p=np.asarray(readscalar(str(work),t,'p',verbose=False))
        geom,tau,wall_p=_wall(work,t,'body')
        area=np.linalg.norm(geom.face_areas,axis=1)
        force=np.sum(wall_p[:,None]*geom.face_areas,axis=0)-np.sum(tau*area[:,None],axis=0)
        errors,diag=ahmed.score(centres,u,p,geom.face_centres,wall_p,force,int(conf['slant_deg']))
        obs.update(diag)
        reference.update(source='Lienhart and Becker 2003, ERCOFTAC case 82',u_ref=ahmed.U_REF)
    elif adapter in ('plate','bump','naca'):
        patch={'plate':'plate','bump':'bump','naca':'aerofoil'}[adapter]
        geom,tau,p=_wall(work,t,patch);area=np.linalg.norm(geom.face_areas,axis=1)
        span=float(np.ptp(foammesh.read_points(work/'constant/polyMesh')[:,2]));sref={'plate':2.,'bump':1.5,'naca':1.}[adapter]
        angle=math.radians(float(conf.get('alpha_deg',0)));drag=np.array([math.cos(angle),math.sin(angle),0.]);lift=np.array([-math.sin(angle),math.cos(angle),0.])
        coefficients=surface_force_coefficients(geom.face_areas,tau,p,span,sref,float(conf.get('alpha_deg',0)))
        normal=geom.face_areas/area[:,None];tangent=drag-np.einsum('ij,j->i',normal,drag)[:,None]*normal;tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-30)
        cf=-2*np.einsum('ij,ij->i',tau,tangent);idx=np.argsort(geom.face_centres[:,0]);x=geom.face_centres[:,0]
        obs.update(**coefficients,
                   wall_x=x[idx].tolist(),wall_y=geom.face_centres[idx,1].tolist(),wall_cf=cf[idx].tolist(),wall_cp=(2*p[idx]).tolist())
        obs['force_history_comparison']=force_history_comparison(work,t,float(conf.get('alpha_deg',0)),coefficients)
        parity=obs['force_history_comparison']
        obs['force_history_parity_passed']=bool(parity['available'] and parity['passed'])
        if parity['available']:
            obs['force_history_parity_error']=max(abs(v) for v in parity['absolute_differences'].values())
        if adapter=='naca':
            table=_numeric_rows(ROOT/conf['force_reference'],3);mask=abs(table[:,0]-float(conf.get('alpha_deg',0)))<=.7
            if not mask.any():raise ValueError('No experimental force group at requested incidence')
            observed=table[mask];cl,cd=observed[:,1:3].mean(axis=0)
            errors={'drag_abs_error':abs(obs['drag_coefficient']-float(cd)),
                    'lift_abs_error':abs(obs['lift_coefficient']-float(cl))}
            reference.update(actual_incidence=observed[:,0].tolist(),force_reference_mean={'cl':float(cl),'cd':float(cd)},
                             sample_count=int(mask.sum()),observed_ranges_not_confidence_intervals=True)
        else:
            table=_numeric_rows(ROOT/conf['cf_reference'],2,first_zone=True);mask=(table[:,0]>=x.min())&(table[:,0]<=x.max())
            if not mask.any():raise ValueError('Reference does not overlap wall coordinates')
            sampled=np.interp(table[mask,0],x[idx],cf[idx]);errors={'skin_friction_rmse':float(np.sqrt(np.mean((sampled-table[mask,1])**2)))}
            reference['numerical_reference_only']=True
    else:raise ValueError(adapter)
    if 'wall_cf' in obs:
        profile=np.asarray(obs['wall_cf'],dtype=float)
        obs.update(wall_cf_rms=float(np.sqrt(np.mean(profile*profile))),
                   wall_cf_mean=float(np.mean(profile)),wall_cf_max_abs=float(np.max(abs(profile))))
    if not errors or not all(np.isfinite(v) and v>=0 for v in errors.values()):
        raise ValueError('Invalid or missing score components')
    return {'errors':errors,'observables':obs,'reference':reference,
            'qualification':'Raw metric extraction only; execution layer decides numerical/constraint eligibility'}
