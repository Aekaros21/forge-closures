"""Explicit development seeds; names imply no gate or performance verdict."""
from tedp.spec import CandidateSpec, Term, struct_hash
from .tuning import default_bounds

ISLANDS = ('I', 'II', 'III', 'IV')
ISLAND_PROFILES = {
    'I': ('ISLAND I - DISCOVERY: production and turbulence-timescale mechanisms for separated and '
          'pressure-gradient flows (hills, hump, curved step, diffuser).\n'
          'You are one of three discovery islands; island IV combines what the discovery islands find. '
          'Your value to the campaign is a mechanism in YOUR area that the other islands are not exploring.\n'
          'Exploit: extend and repair this island\'s own best lineage (shown as ISLAND FRONTIER). '
          'Explore: introduce a production/timescale mechanism new to this island (a different activation region, '
          'invariant or tensor route), not a variant of another island\'s structure.\n'
          'Hold attached low-Re wall Cf within allowance; broad near-wall production sources have failed there. '
          'Do not make a candidate near-inert merely to survive admission.'),
    'II': ('ISLAND II - DISCOVERY: constitutive anisotropy, normal-stress and secondary-flow mechanisms.\n'
           'You are one of three discovery islands; island IV combines what the discovery islands find. '
           'Your value to the campaign is a stress mechanism in YOUR area that the other islands are not exploring.\n'
           'Exploit: extend and repair this island\'s own best lineage (shown as ISLAND FRONTIER). '
           'Explore: introduce a constitutive mechanism new to this island (a different tensor, basis combination or '
           'amplitude law), aimed where a stress mechanism is physically warranted, not a variant of another island\'s structure.\n'
           'Use bounded, amplitude-aware normalization; realizability failures have come from unnormalized T2/T3/T4 sums.'),
    'III': ('ISLAND III - DISCOVERY: activation, pressure-gradient context and rotation/curvature response within '
            'the implemented input grammar.\n'
            'You are one of three discovery islands; island IV combines what the discovery islands find. '
            'Your value to the campaign is a rotation, curvature or activation mechanism the other islands are not exploring.\n'
            'Exploit: extend and repair this island\'s own best lineage (shown as ISLAND FRONTIER). '
            'Explore: introduce a response mechanism new to this island (a different sensor of rotation, curvature or '
            'non-equilibrium), not a variant of another island\'s structure.\n'
            'Hold hill and hump Cf within allowance; equilibrium-weighted signed sources have regressed there.'),
    'IV': ('ISLAND IV - COMBINATION: construct feasible combinations of mechanisms that the discovery islands '
           '(I production/timescale, II constitutive stress, III rotation/activation) have shown in CFD.\n'
           'The DISCOVERY ISLAND LEADERS block lists each discovery island\'s best mechanism. '
           'Exploit: combine proven mechanisms from different discovery islands so their gains add, preserving the '
           'mechanism behind the largest family gain. Explore: try a new pairing or a new way of partitioning the '
           'activation regions of discovery-island mechanisms.\n'
           'Prefer mechanisms whose activation regions are physically complementary. '
           'Do not merely shrink amplitudes of an existing coupled model.'),
}
DISCOVERY_ISLANDS = ('I', 'II', 'III')
COMBINATION_ISLAND = 'IV'


def default_seeds() -> dict[str, list[CandidateSpec]]:
    gate = '(max(PoE-1.0,0.0)/max(1.0,PoE))'
    topology = 'tanh(c1*abs(I1+I2)/(I1+abs(I2)))'
    return {
        'I': [CandidateSpec('production_topology_seed', rsource=(
            Term('T1', f'c0*(1-F1)*{topology}/max(1.0,c1*I1)'),), constants=(.1, 25.))],
        'II': [CandidateSpec('anisotropy_seed', bdelta=(
            Term('T2', f'(c0/c1)*{gate}/max(1.0,c1*sqrt(max(I1*abs(I2),0.0)))'),), constants=(.1, 25.))],
        'III': [CandidateSpec('production_deficit_seed', rsource=(
            Term('T1', f'c0*{gate}*{topology}/max(1.0,c1*I1)'),), constants=(.1, 25.))],
        'IV': [CandidateSpec('frozen_GNE_SST_incumbent', bdelta=(
            Term('T2', f'(c0/c1)*{gate}/max(1.0,c1*sqrt(max(I1*abs(I2),0.0)))'),
            Term('T3', f'(c0/c1)*(1-F1)*{gate}/max(1.0,c1*I1)'),),
            rsource=(Term('T1', f'c0*{gate}*{topology}/max(1.0,c1*I1)'),), constants=(4., 25.))],
    }


def seed_parameter_types(spec: CandidateSpec) -> list[str] | None:
    known = {struct_hash(candidate) for group in default_seeds().values() for candidate in group}
    return ['signed', 'positive'] if struct_hash(spec) in known else None


def seed_bounds(spec: CandidateSpec):
    return default_bounds(spec, seed_parameter_types(spec))
