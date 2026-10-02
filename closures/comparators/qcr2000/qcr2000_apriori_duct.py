"""A-priori QCR2000 spec on the converged SST duct field (squareDuct_Re_2000): realizability-check
activity, gMax activity, and the sign of the streamwise-vorticity source against the DNS vorticity.

Read-only post-processing of a converged SST solution of squareDuct_Re_2000 at iteration 5000 (in the paper,
the SST run of the holdout verification); no CFD. Gradients by np.gradient on the 96 x 96 cell-centre grid.
Run (the case from reproduce/make_case.py squareDuct_Re_2000 sst and its Allrun, the DNS from
data/fetch_data.py ducts):
    python3 closures/comparators/qcr2000/qcr2000_apriori_duct.py --case runs/squareDuct_Re_2000_sst \
        --reference data/mcconkey/cache/holdout_duct_squareDuct_Re_2000.npz --out qcr2000_apriori_duct.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import fluidfoam as ff
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]/'evaluation'), str(HERE)]
import qcr2000_comparator as Q

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--case', required=True, help='converged SST case of squareDuct_Re_2000 (constant/polyMesh, 5000/)')
ap.add_argument('--reference', default=str(HERE.parents[2]/'data/mcconkey/cache/holdout_duct_squareDuct_Re_2000.npz'))
ap.add_argument('--out', default='qcr2000_apriori_duct.json')
args = ap.parse_args()
case = args.case
x, y, z = ff.readmesh(case, verbose=False)
U = ff.readvector(case, '5000', 'U', verbose=False)
k = ff.readscalar(case, '5000', 'k', verbose=False)
om = ff.readscalar(case, '5000', 'omega', verbose=False)
nut = ff.readscalar(case, '5000', 'nut', verbose=False)
n = len(k)
yy = np.unique(np.round(y, 12)); zz = np.unique(np.round(z, 12))
ny, nz = len(yy), len(zz)
assert ny * nz == n, (ny, nz, n)
iy = np.searchsorted(yy, np.round(y, 12)); iz = np.searchsorted(zz, np.round(z, 12))
def grid(f):
    g = np.empty((ny, nz)); g[iy, iz] = f; return g
Ug = [grid(U[i]) for i in range(3)]
kg, omg, nutg = grid(k), grid(om), grid(nut)
grad = np.zeros((ny, nz, 3, 3))           # dU_i/dx_j, x streamwise homogeneous
for i in range(3):
    grad[:, :, i, 1] = np.gradient(Ug[i], yy, axis=0)
    grad[:, :, i, 2] = np.gradient(Ug[i], zz, axis=1)
spec = Q.qcr2000_spec()
G = grad.reshape(-1, 3, 3); K = kg.ravel(); OM = omg.ravel(); NUT = nutg.ravel()
tau, var, g2 = Q.spec_stress(spec, G, K, NUT, OM, np.zeros_like(K), np.zeros((len(K), 3)))
ref = Q.qcr2000_reference(G, K, NUT, np.zeros((len(K), 3)))
rel = Q.relative_difference(tau, ref)
# realizability check of the solver: eigenvalues of -nut/k dev(S) + b in [-1/3-1e-6, 2/3+1e-6]
S = 0.5 * (G + np.swapaxes(G, 1, 2)); Sd = S - np.trace(S, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
bB = -(NUT / K)[:, None, None] * Sd
bD = tau / (2 * K)[:, None, None]
ev = np.linalg.eigvalsh(bB + bD); evB = np.linalg.eigvalsh(bB)
ok = (ev[:, 0] >= -1/3 - 1e-6) & (ev[:, -1] <= 2/3 + 1e-6)
okB = (evB[:, 0] >= -1/3 - 1e-6) & (evB[:, -1] <= 2/3 + 1e-6)
# streamwise vorticity source of the added stress: d2(Ryy-Rzz)/dydz + (d2/dz2 - d2/dy2) Ryz
T = tau.reshape(ny, nz, 3, 3)
dyy_zz = T[:, :, 1, 1] - T[:, :, 2, 2]; ryz = T[:, :, 1, 2]
src = (np.gradient(np.gradient(dyy_zz, yy, axis=0), zz, axis=1)
       + np.gradient(np.gradient(ryz, zz, axis=1), zz, axis=1)
       - np.gradient(np.gradient(ryz, yy, axis=0), yy, axis=0))
# DNS streamwise vorticity on the same grid
from scipy.interpolate import griddata
d = np.load(args.reference)
P = d['points']; ud = d['u']
Y, Z = np.meshgrid(yy, zz, indexing='ij')
vd = griddata(P, ud[:, 1], (Y, Z), method='linear'); wd = griddata(P, ud[:, 2], (Y, Z), method='linear')
oxd = np.gradient(wd, yy, axis=0) - np.gradient(vd, zz, axis=1)
# region: 8 triangles (each quadrant split by its corner bisector), excluding the first 3 cells from walls
inner = np.zeros_like(Y, bool); inner[3:-3, 3:-3] = True
res = {}
for sy in (-1, 1):
    for sz in (-1, 1):
        q = (np.sign(Y) == sy) & (np.sign(Z) == sz) & inner
        below = q & (np.abs(Z) > np.abs(Y))    # nearer the z = +-0.5 wall
        above = q & (np.abs(Z) < np.abs(Y))
        for name, m in (('near_z_wall', below), ('near_y_wall', above)):
            good = m & np.isfinite(oxd)
            res[f'{sy:+d}{sz:+d}_{name}'] = {
                'mean_source_sign': float(np.sign(np.mean(src[good]))),
                'dns_vorticity_sign': float(np.sign(np.mean(oxd[good]))),
                'area_fraction_sign_agree': float(np.mean(np.sign(src[good]) == np.sign(oxd[good]))),
            }
agree = [v['mean_source_sign'] == v['dns_vorticity_sign'] for v in res.values()]
corr = float(np.nansum(src * oxd * inner) / np.sqrt(np.nansum((src * inner) ** 2) * np.nansum((oxd * inner) ** 2)))
out = {
    'field': case + '/5000', 'cells': n,
    'spec_vs_numpy_qcr_max_relative_difference_on_field': float(rel[np.isfinite(rel)].max()),
    'realizability_check_fails_with_qcr_fraction': float(1 - ok.mean()),
    'realizability_check_fails_sst_alone_fraction': float(1 - okB.mean()),
    'g2_abs_max': float(np.abs(g2).max()),
    'pi_max': float(var['PoE'].max()),
    'vorticity_source_vs_dns_vorticity_triangles': res,
    'triangles_with_matching_sign': int(sum(agree)),
    'source_dns_vorticity_correlation_interior': corr,
    'trace_fraction_np_gradient_max': float(np.max(np.abs(np.trace(G, axis1=1, axis2=2)) / np.maximum(np.sqrt(np.einsum('nij,nij->n', G, G)), 1e-300))),
}
out['g2_abs_gt_10_cell_fraction'] = float(np.mean(np.abs(g2) > 10.0))
gm = np.clip(g2, -10, 10)
tau10 = 2 * K[:, None, None] * gm[:, None, None] * (tau / np.where(g2 == 0, 1, 2 * K * g2)[:, None, None])
out['stress_rms_change_if_gmax_10'] = float(np.sqrt(np.sum((tau10 - tau) ** 2)) / np.sqrt(np.sum(tau ** 2)))
print(json.dumps(out, indent=1))
json.dump(out, open(args.out, 'w'), indent=1)
