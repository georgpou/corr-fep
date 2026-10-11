#!/usr/bin/env python3
"""Experimental environment-only water conditioning for the author MixedMAF.

The three *spatial sites* are fixed using one training frame. At evaluation time
the matching uses only unchanged environment-water oxygen coordinates, never
ligand coordinates or trajectory indices. This is a separate candidate, NOT a
replacement for the published ligand-aligned matching algorithm.
"""
import numpy as np
import scipy.optimize
import torch
import tfep.app.mixedmaf
from tfep.utils.misc import atom_to_flattened_indices

from stage_b_author_map import AuthorMixedMAF
from stage_b_author_issue_audit import load_driver


def assignment_from_environment(oxygen_positions, template_sites):
    """Assign 3 distinct physical waters to 3 frozen spatial sites.

    Coordinates are the existing, ligand-centered/box-wrapped representation.
    The per-sample rule does not read the ligand coordinates. Using Cartesian
    (not minimum-image) distances avoids gathering discontinuous PBC images
    into the Cartesian reference geometry.
    """
    water = np.asarray(oxygen_positions, dtype=np.float64)
    sites = np.asarray(template_sites, dtype=np.float64)
    if sites.shape != (3, 3) or water.ndim != 2 or water.shape[1] != 3 or len(water) < 3:
        raise ValueError('Expected three sites and at least three water oxygens')
    delta = sites[:, None, :] - water[None, :, :]
    costs = np.sum(delta * delta, axis=-1)
    rows, cols = scipy.optimize.linear_sum_assignment(costs)
    if not np.array_equal(rows, np.arange(3)):
        raise AssertionError('Unexpected site ordering from water assignment')
    return np.asarray(cols, dtype=np.int64)


def environmental_sites(u, legacy_preprocessing, reference_frame):
    """Freeze original author's three *reference atom indices*, but match new waters."""
    load_driver().fix_universe_elements(u)
    refs = np.asarray(legacy_preprocessing['references'], dtype=np.int64)
    if len(refs) != 3 or len(set(refs)) != 3:
        raise AssertionError('Exactly three distinct author reference sites required')
    oxygens = u.select_atoms('resname SOL and element O').indices.astype(np.int64)
    if not set(refs).issubset(set(oxygens)):
        raise AssertionError('Author selected non-water-oxygen reference; not supported by this probe')
    u.trajectory[int(reference_frame)]
    sites = np.asarray(u.atoms.positions[refs], dtype=np.float64).copy()
    if not np.isfinite(sites).all():
        raise AssertionError('Nonfinite training reference sites')
    return dict(references=refs.tolist(), reference_frame=int(reference_frame),
                candidate_oxygens=oxygens, template_sites=sites)


class EnvironmentOnlyContextFlow(torch.nn.Module):
    """Conditional map T(x,e)=(F_{A(e)}(x;e),e); assignment is recomputed from e."""
    def __init__(self, flow, context, environment, return_partial=False):
        super().__init__()
        self.flow = flow
        self.return_partial = return_partial
        self.register_buffer('reference_dofs', atom_to_flattened_indices(
            torch.as_tensor(context['references'], dtype=torch.long)))
        self.register_buffer('environment_dofs', atom_to_flattened_indices(
            torch.as_tensor(environment, dtype=torch.long)))
        self.register_buffer('candidate_oxygens', torch.as_tensor(
            context['candidate_oxygens'], dtype=torch.long))
        self.register_buffer('template_sites', torch.as_tensor(
            context['template_sites'], dtype=torch.float64))

    def assignments(self, x):
        # Selection is discrete by design; ligand gradients still flow through
        # the mapped coordinates after gathering environment reference atoms.
        env_xyz = x.detach().reshape(len(x), -1, 3)[:, self.candidate_oxygens].cpu().numpy()
        sites = self.template_sites.cpu().numpy()
        candidate = self.candidate_oxygens.cpu().numpy()
        physical = np.stack([candidate[assignment_from_environment(water, sites)]
                             for water in env_xyz])
        return torch.as_tensor(physical, dtype=torch.long, device=x.device)

    def _pass(self, x, inverse):
        physical = self.assignments(x)
        dofs = atom_to_flattened_indices(physical)
        if dofs.ndim != 2 or dofs.shape != (len(x), len(self.reference_dofs)):
            raise AssertionError('Environment assignment has wrong shape')
        context = x.clone()
        context[:, self.reference_dofs] = torch.gather(x, dim=1, index=dofs)
        result = self.flow.inverse(context) if inverse else self.flow(context)
        if self.return_partial:
            return result
        y = result[0].clone()
        y[:, self.environment_dofs] = x[:, self.environment_dofs]
        return (y, *result[1:])

    def forward(self, x, ignored_sample_index=None):
        return self._pass(x, False)

    def inverse(self, y, ignored_sample_index=None):
        return self._pass(y, True)


class EnvironmentOnlyMixedMAF(AuthorMixedMAF):
    """Same six-layer upstream MixedMAF, with an environment-only reference rule."""
    def create_partial_flow(self, flow, return_partial=False):
        partial = tfep.app.mixedmaf.MixedMAFMap.create_partial_flow(self, flow, return_partial)
        environment = np.setdiff1d(np.arange(self.dataset.n_atoms), self.ligand_idx)
        return EnvironmentOnlyContextFlow(partial, self.preprocessing, environment, return_partial)

    def forward(self, batch):
        y, logj = self._flow(batch['positions'])
        return dict(positions=y, log_det_J=logj)

    def inverse(self, batch):
        y, logj = self._flow.inverse(batch['positions'])
        return dict(positions=y, log_det_J=logj)
