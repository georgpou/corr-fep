#!/usr/bin/env python3
"""Experimental environment-only water conditioning for the author MixedMAF.

The three *spatial sites* are fixed using one training frame. At evaluation time
the matching uses an unchanged environment anchor and periodic minimum-image
water offsets, never ligand coordinates or trajectory indices. This is a separate candidate, NOT a
replacement for the published ligand-aligned matching algorithm.
"""
import numpy as np
import scipy.optimize
import torch
import tfep.app.mixedmaf
from tfep.utils.misc import atom_to_flattened_indices

from stage_b_author_map import AuthorMixedMAF
from stage_b_author_issue_audit import load_driver


def assignment_from_environment(oxygen_positions, template_sites, anchor_position, box_lengths):
    """Assign 3 distinct physical waters to 3 frozen spatial sites.

    Use periodic water displacements relative to one fixed environment anchor.
    This is invariant to global translation and to ligand-dependent water image
    wrapping, provided the box is orthorhombic and transformations preserve the
    physical environment configuration.
    """
    water = np.asarray(oxygen_positions, dtype=np.float64)
    sites = np.asarray(template_sites, dtype=np.float64)
    if sites.shape != (3, 3) or water.ndim != 2 or water.shape[1] != 3 or len(water) < 3:
        raise ValueError('Expected three sites and at least three water oxygens')
    box = np.asarray(box_lengths, dtype=np.float64)
    anchor = np.asarray(anchor_position, dtype=np.float64)
    if box.shape != (3,) or np.any(box <= 0) or anchor.shape != (3,):
        raise ValueError('Expected positive orthorhombic box lengths and one anchor')
    displacement = water - anchor
    displacement -= box * np.rint(displacement / box)
    delta = sites[:, None, :] - displacement[None, :, :]
    delta -= box * np.rint(delta / box)
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
    nonsolvent = u.select_atoms('not resname MOL and not resname SOL and not element H').indices
    # An isolated ligand-in-water box need not contain a non-solvent atom.
    # Any *fixed physical environment atom* is a valid mathematical anchor.
    # A water anchor may be physically noisy and must be evaluated separately.
    anchor = int(nonsolvent[0]) if len(nonsolvent) else int(oxygens[0])
    box = u.dimensions.copy()
    if not np.allclose(box[3:], 90., atol=1e-5):
        raise AssertionError('Orthorhombic cell required for this prototype')
    sites = np.asarray(u.atoms.positions[refs] - u.atoms.positions[anchor], dtype=np.float64)
    sites -= box[:3] * np.rint(sites / box[:3])
    if not np.isfinite(sites).all():
        raise AssertionError('Nonfinite training reference sites')
    return dict(references=refs.tolist(), reference_frame=int(reference_frame),
                candidate_oxygens=oxygens, template_sites=sites, anchor_atom=anchor,
                anchor_kind='nonsolvent' if len(nonsolvent) else 'water_oxygen')


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
        self.register_buffer('anchor_atom', torch.as_tensor(context['anchor_atom'], dtype=torch.long))
        self.register_buffer('template_sites', torch.as_tensor(
            context['template_sites'], dtype=torch.float64))

    def assignments(self, x, box_lengths):
        # Selection is discrete by design; ligand gradients still flow through
        # the mapped coordinates after gathering environment reference atoms.
        env_xyz = x.detach().reshape(len(x), -1, 3)[:, self.candidate_oxygens].cpu().numpy()
        sites = self.template_sites.cpu().numpy()
        candidate = self.candidate_oxygens.cpu().numpy()
        anchor = x.detach().reshape(len(x), -1, 3)[:, self.anchor_atom].cpu().numpy()
        boxes = torch.as_tensor(box_lengths).detach().cpu().numpy()
        physical = np.stack([candidate[assignment_from_environment(water, sites, a, box)]
                             for water, a, box in zip(env_xyz, anchor, boxes)])
        return torch.as_tensor(physical, dtype=torch.long, device=x.device)

    def _pass(self, x, inverse, box_lengths):
        physical = self.assignments(x, box_lengths)
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

    def forward(self, x, box_lengths):
        return self._pass(x, False, box_lengths)

    def inverse(self, y, box_lengths):
        return self._pass(y, True, box_lengths)


class EnvironmentOnlyMixedMAF(AuthorMixedMAF):
    """Same six-layer upstream MixedMAF, with an environment-only reference rule."""
    def create_partial_flow(self, flow, return_partial=False):
        partial = tfep.app.mixedmaf.MixedMAFMap.create_partial_flow(self, flow, return_partial)
        environment = np.setdiff1d(np.arange(self.dataset.n_atoms), self.ligand_idx)
        return EnvironmentOnlyContextFlow(partial, self.preprocessing, environment, return_partial)

    def forward(self, batch):
        # TFEP's spline calibration passes 'dimensions'; physical pilot uses 'cell'.
        cell = batch['cell'] if 'cell' in batch else batch['dimensions']
        y, logj = self._flow(batch['positions'], cell[:, :3])
        return dict(positions=y, log_det_J=logj)

    def inverse(self, batch):
        cell = batch['cell'] if 'cell' in batch else batch['dimensions']
        y, logj = self._flow.inverse(batch['positions'], cell[:, :3])
        return dict(positions=y, log_det_J=logj)
