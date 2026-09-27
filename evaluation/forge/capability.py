"""Measured offline capability diagnostics, never a substitute for CFD."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tedp import candidate, tensors
from tedp.holdout_cases.duct_scoring import read_internal_field
from .seeds import default_seeds


def projection_diagnostic(basis, target):
    """Per-state unrestricted least squares: optimistic span diagnostic only."""
    basis=np.asarray(basis);target=np.asarray(target)
    if basis.ndim!=4 or target.shape!=(len(basis),3,3):
        raise ValueError('Expected (N,K,3,3) basis and (N,3,3) target')
    if not np.isfinite(basis).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite projection inputs')
    ranks=[];conditions=[];residual=[]
    for terms,b in zip(basis,target):
        a=terms.reshape(len(terms),9).T
        scale=np.linalg.norm(a,axis=0);a=a/np.where(scale>1e-30,scale,1)
        coeff,_,rank,s=np.linalg.lstsq(a,b.reshape(9),rcond=1e-10)
        ranks.append(int(rank));residual.append(float(np.sum((a@coeff-b.reshape(9))**2)))
        if rank:conditions.append(float(s[0]/s[rank-1]))
    denom=float(np.sum(target*target))
    return {'samples':len(target),'rank_histogram':{str(i):ranks.count(i) for i in sorted(set(ranks))},
        'normalized_column_condition_median':float(np.median(conditions)) if conditions else None,
        'target_relative_frobenius_residual':float(np.sqrt(sum(residual)/max(denom,1e-30))),
        'interpretation':'Independent coefficients at each sampled cell; an optimistic span diagnostic, not an invariant learned function or CFD prediction'}


def ambiguity_diagnostic(features, target, *, feature_radius=.02, target_tolerance=.02):
    """Nearby invariant inputs versus differing invariant target eigenvalues.

    Raw tensor components are deliberately not compared across orientations.
    Numerical/reference uncertainty can also produce disagreement; this is not
    proof that extra inputs or a transport model are necessary.
    """
    features=np.asarray(features);target=np.asarray(target)
    if len(features)<2:return {'status':'insufficient_samples'}
    scale=np.maximum(np.std(features,axis=0),1e-12);x=(features-np.mean(features,axis=0))/scale
    distances,indices=cKDTree(x).query(x,k=2)
    # Duplicate feature vectors can put a different sample before the query
    # itself. Choose an explicitly non-self neighbor rather than column one.
    column=np.where(indices[:,0]!=np.arange(len(x)),0,1)
    neighbor=indices[np.arange(len(x)),column]
    near=distances[np.arange(len(x)),column]<=feature_radius
    eigen=np.linalg.eigvalsh(target);delta=np.linalg.norm(eigen-eigen[neighbor],axis=1)
    return {'status':'diagnostic_complete','feature_radius_in_standardized_coordinates':feature_radius,
        'target_eigenvalue_tolerance':target_tolerance,'samples':len(x),'near_neighbors':int(near.sum()),
        'near_neighbors_with_target_disagreement':int(np.sum(near&(delta>target_tolerance))),
        'near_target_difference_median':float(np.median(delta[near])) if near.any() else None,
        'interpretation':'Potential input ambiguity on this sample; orientation-invariant eigenvalues, not a proof of irreducible error'}


def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sym(table,prefix):
    b=np.zeros((len(table[prefix+'11']),3,3))
    for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]:
        b[:,i,j]=b[:,j,i]=table[prefix+str(i+1)+str(j+1)]
    return b


def _mcconkey(root,case,max_samples=1024):
    cid=case['adapter_config']['case_id'];cache=root/'data/mcconkey/cache'
    sp,rp=cache/f'komegasst_{cid}.npz',cache/f'REF_{cid}.npz'
    with np.load(sp) as archive:s={k:archive[k] for k in archive.files}
    with np.load(rp) as archive:r={k:archive[k] for k in archive.files}
    grad=np.stack([s[f'komegasst_gradU_{i}{j}'] for i in (1,2,3) for j in (1,2,3)],axis=1).reshape(-1,3,3)
    k=s['komegasst_k'];omega=s['komegasst_omega'];nut=s['komegasst_nut'];y=s['komegasst_wallDistance']
    reference_b=_sym(r,'REF_b_')
    valid=np.isfinite(grad).all(axis=(1,2))&(k>0)&(omega>0)&(y>0)
    physical_valid=valid&np.isfinite(k)&np.isfinite(omega)&np.isfinite(nut)&np.isfinite(y)
    reference_valid=np.isfinite(reference_b).all(axis=(1,2))
    gradient_valid=np.ones(len(k),dtype=bool)
    for prefix in ('komegasst_gradk_','komegasst_gradomega_'):
        for i in (1,2,3):gradient_valid&=np.isfinite(s[prefix+str(i)])
    valid=physical_valid&reference_valid&gradient_valid
    ids=np.flatnonzero(valid)
    if not len(ids):raise ValueError('No finite physical input states')
    ids=ids[np.linspace(0,len(ids)-1,min(max_samples,len(ids)),dtype=int)]
    grad,k,omega,nut,y=grad[ids],k[ids],omega[ids],nut[ids],y[ids]
    sym=.5*(grad+grad.transpose(0,2,1));sym-=np.trace(sym,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3
    shat=sym/omega[:,None,None];what=.5*(grad-grad.transpose(0,2,1))/omega[:,None,None]
    basis,inv=tensors.integrity_basis(shat,what)
    target=reference_b[ids]+(nut/k)[:,None,None]*sym
    production=2*nut*np.sum(sym*sym,axis=(1,2));poe=np.clip(production/(.09*k*omega),0,10)
    template=root/'data/mcconkey/foam/komegasst'/case['family'].split('_',1)[1]/cid
    import re
    nu=float(re.search(r'nu\s+(?:\[[^\]]*\]\s*)?([0-9.eE+-]+)\s*;', (template/'constant/transportProperties').read_text()).group(1))
    gk=np.stack([s[f'komegasst_gradk_{i}'][ids] for i in (1,2,3)],axis=1)
    gw=np.stack([s[f'komegasst_gradomega_{i}'][ids] for i in (1,2,3)],axis=1)
    cd=np.maximum(2*.856*np.sum(gk*gw,axis=1)/omega,1e-10)
    arg=np.minimum(np.maximum(np.sqrt(k)/(.09*omega*y),500*nu/(y*y*omega)),4*.856*k/(cd*y*y))
    f1=np.tanh(np.minimum(arg,100)**4);ret=k/(nu*omega)
    states=candidate.make_states(shat,what,ret,f1,poe)
    seed=default_seeds()['IV'][0]
    stress=candidate.eval_bdelta(seed,states);source=candidate.eval_rhat_raw(seed,states)
    features=np.column_stack([np.arcsinh(inv),np.log1p(ret),f1,poe])
    return {'status':'diagnostic_complete','source_hashes':{str(p.relative_to(root)):_sha(p) for p in (sp,rp)},
        'sample_indices':ids.tolist(),'excluded_states':int((~valid).sum()),
        'excluded_nonphysical_input_states':int((~physical_valid).sum()),
        'excluded_nonfinite_reference_states':int((~reference_valid).sum()),
        'excluded_nonfinite_turbulence_gradient_states':int((~gradient_valid).sum()),
        'inputs':'Stored SST gradients/k/omega/nut; reconstructed SST F1/PoE/Ret, not a fresh coupled solution',
        'target':'Reference anisotropy minus stored SST linear anisotropy at matching cells',
        'projection':{name:projection_diagnostic(basis[:,indices],target) for name,indices in
                      [('T1',[0]),('T1_T3',[0,1,2]),('T1_T10',list(range(10)))]},
        'input_ambiguity':ambiguity_diagnostic(features,reference_b[ids]),
        'frozen_gne_activation':{'q_positive_fraction':float(np.mean(poe>1)),
            'stress_nonzero_fraction':float(np.mean(np.linalg.norm(stress,axis=(1,2))>1e-12)),
            'source_nonzero_fraction':float(np.mean(abs(source)>1e-12))}}


def _duct(root,case,max_samples=1024):
    conf=case['adapter_config'];npz=root/conf['reference_npz'];tau_file=root/conf['dns_case']/'0/tau'
    with np.load(npz) as a:points=a['points'];u=a['u']
    tau=read_internal_field(tau_file,6);n=len(points)
    if tau.shape!=(n,6):raise ValueError('Duct full-stress/reference ordering mismatch')
    u_file=tau_file.parent/'U'
    source_u=read_internal_field(u_file,3)
    if source_u.shape!=u.shape or not np.allclose(source_u,u,rtol=1e-12,atol=1e-14):
        raise ValueError('Duct full-stress source and cached DNS velocity ordering differ')
    yy,zz=np.unique(points[:,0]),np.unique(points[:,1])
    if len(yy)*len(zz)!=n:raise ValueError('Duct oracle needs the existing tensor-product DNS cross-section')
    iy,iz=np.searchsorted(yy,points[:,0]),np.searchsorted(zz,points[:,1]);field=np.zeros((len(yy),len(zz),3));field[iy,iz]=u
    gy,gz=np.gradient(field,yy,zz,axis=(0,1));g=np.zeros((n,3,3));g[:,:,1]=gy[iy,iz];g[:,:,2]=gz[iy,iz]
    s=.5*(g+g.transpose(0,2,1));s-=np.trace(s,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3;w=.5*(g-g.transpose(0,2,1))
    scale=np.maximum(np.sqrt(np.sum(s*s+w*w,axis=(1,2))),1e-12);basis,_=tensors.integrity_basis(s/scale[:,None,None],w/scale[:,None,None])
    trace=tau[:,0]+tau[:,3]+tau[:,5];valid=trace>1e-12;ids=np.flatnonzero(valid);ids=ids[np.linspace(0,len(ids)-1,min(max_samples,len(ids)),dtype=int)]
    b=np.zeros((n,3,3))
    for column,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):b[:,i,j]=b[:,j,i]=tau[:,column]/np.maximum(trace,1e-30)
    b-=np.eye(3)/3
    return {'status':'diagnostic_complete','source_hashes':{str(p.relative_to(root)):_sha(p) for p in (npz,tau_file,u_file)},
        'scope':'DNS-gradient oracle basis span only; DNS gradients are not deployed SST inputs. Finite differences add discretization error.',
        'sample_indices':ids.tolist(),'normal_stress_difference_rms':float(np.sqrt(np.mean((tau[:,3]-tau[:,5])**2))),
        'projection':{name:projection_diagnostic(basis[ids][:,indices],b[ids]) for name,indices in
                      [('T1',[0]),('T1_T3',[0,1,2]),('T1_T10',list(range(10)))]}}


def canonical_gate_probe():
    gradients=np.array([[[0,1,0],[0,0,0],[0,0,0]],[[0,1,0],[0,0,0],[0,0,0]],
                        [[1,0,0],[0,-1,0],[0,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]],float)
    s=.5*(gradients+gradients.transpose(0,2,1));w=.5*(gradients-gradients.transpose(0,2,1))
    states=candidate.make_states(s,w,np.ones(4),np.zeros(4),np.array([1,2,2,2.]))
    spec=default_seeds()['IV'][0]
    return {'labels':['equilibrium_shear','nonequilibrium_shear','pure_strain','pure_rotation'],
        'poe':states.poe.tolist(),'invariants':states.inv.tolist(),
        'gne_stress_norm':np.linalg.norm(candidate.eval_bdelta(spec,states),axis=(1,2)).tolist(),
        'gne_source_rhat':candidate.eval_rhat_raw(spec,states).tolist(),
        'scope':'Canonical analytical/Python gate response; not CFD capability or performance'}


def run_pilot(root, manifest, output, *, cfd_callback=None, resume=False):
    root=Path(root).resolve();output=Path(output).absolute()
    if not output.resolve().is_relative_to(root/'reports'):
        raise ValueError('Pilot output must be inside v2 reports/')
    previous=None
    if output.exists():
        if not resume:raise FileExistsError(f'Refusing to replace capability evidence: {output}')
        previous=json.loads(output.read_text())
    results={};chosen_mc=set()
    for case in manifest['cases']:
        if case['role']=='final':continue
        adapter=case['adapter']
        if adapter=='mcconkey' and case['family'] not in chosen_mc:
            chosen_mc.add(case['family'])
            try:results[case['id']]=_mcconkey(root,case)
            except (OSError,ValueError,KeyError) as exc:results[case['id']]={'status':'unavailable','reason':str(exc)}
        elif adapter=='duct' and not any(k.startswith('squareDuct_') for k in results):
            try:results[case['id']]=_duct(root,case)
            except (OSError,ValueError,KeyError) as exc:results[case['id']]={'status':'unavailable','reason':str(exc)}
    report={'kind':'offline_capability_diagnostics','status':'complete' if results and all(r['status']=='diagnostic_complete' for r in results.values()) else 'partial',
            'case_diagnostics':results,'canonical_gate_probe':canonical_gate_probe(),
            'cfd_capability':{'status':'not_evaluated','passed':False,
                  'reason':'No coupled perturbation/response or aerodynamic protection CFD is implied by offline projection'},
            'release_ready':False}
    if cfd_callback is not None:
        report['cfd_capability']=cfd_callback(root,manifest)
    if previous is not None:
        if previous!=report:
            raise ValueError('Existing capability evidence differs from current diagnostics; use a new campaign')
        return previous
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
