#!/usr/bin/env python3
"""V236 online proposal with generating-PDF weights; no optimizer changes.

Inherits the verified V235 objective, gradient diagnostics and L-BFGS weighting.
Only joint-point acquisition changes from fixed donor replay to model-dependent
proposal evaluation. Transition points and state must come from V235.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from objective_control_v235 import ObjectiveContext, inverse_density_weights, sha_points
from nonuniform_proposal_v234 import AreaResidualAdaptive2D
from shared_transition_v235 import SharedTransition, file_sha
import convection_diffusion_pilot_v233 as base


class ReadOnlyTransition(SharedTransition):
    """Never create or alter the user's original shared transition."""
    def __init__(self, folder, device, expected_file_sha):
        folder=Path(folder)
        if not (folder/'manifest.json').is_file() or not (folder/'common_transition.pt').is_file():
            raise FileNotFoundError('Original V235 shared transition is missing: '+str(folder))
        meta=json.loads((folder/'manifest.json').read_text())
        if file_sha(folder/'common_transition.pt') != expected_file_sha:
            raise RuntimeError('Shared transition does not match the frozen replay result: '+str(folder))
        super().__init__(folder,meta['signature'],device,'read_only_online_branch')
        self.expected_file_sha=expected_file_sha
        self.meta_sha=file_sha(self.meta_path)

    def begin(self, model, optimizer, initial_state_hash):
        if not self.state_path.is_file() or not self.meta_path.is_file():
            raise FileNotFoundError('Refusing to regenerate a missing shared transition')
        b=super().begin(model,optimizer,initial_state_hash)
        if b is None: raise RuntimeError('Read-only transition unexpectedly attempted creation')
        return b

    def finish(self, *args, **kwargs):
        if self.bundle is None:
            raise RuntimeError('A restored V235 bundle is required before joint training')
        result=super().finish(*args,**kwargs)
        if file_sha(self.state_path)!=self.expected_file_sha or file_sha(self.meta_path)!=self.meta_sha:
            raise RuntimeError('Source shared transition was modified unexpectedly')
        result['training_prefix_policy']='load original V235 model+Adam+RNG; no prefix training'
        result['source_transition_read_only']=True
        return result


class OnlineAreaProposal(AreaResidualAdaptive2D):
    """Same density law/transform as V234, also retain the pre-rounding cell PDF.

    q = normalized cell mass / cell area, evaluated in the GENERATING cell.
    Mapping rounded float32 coordinates back into cells can choose the wrong
    side of a fine cell boundary; therefore indices are saved during inversion.
    """
    def transform(self, uv):
        uv=np.asarray(uv,dtype=np.float64)
        if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all() or (uv<0).any() or (uv>=1).any():
            raise ValueError('Probability coordinates must be finite in [0,1)')
        pt=self.prob.sum(axis=1)
        it,t=self._invert(uv[:,0],pt,self.t_edges)
        x=np.empty(len(uv),dtype=np.float64)
        ix=np.empty(len(uv),dtype=np.int64)
        for j in np.unique(it):
            mask=it==j
            ix[mask],x[mask]=self._invert(uv[mask,1],self.prob[j]/pt[j],self.x_edges)
        self.last_generation_cell=np.column_stack((it,ix))
        self.last_generation_pdf=(self.prob[it,ix]/self.area[it,ix]).copy()
        self.last_pre_rounding_points=np.column_stack((x,t))
        return self.last_pre_rounding_points.astype(np.float32)


class OnlineObjectiveContext(ObjectiveContext):
    def __init__(self, tape_file, cfg, gradient_audit=True):
        # Donor data supply only the unchanged TRANSITION design. They do not
        # supply any of the joint-training probability maps or points below.
        super().__init__(tape_file,'importance_uniform',cfg,gradient_audit)
        self.online_points=[];self.online_pdfs=[];self.online_weights=[]
        self.online_uv=[];self.online_cells=[];self.refresh_metadata=[]

    def make_proposal(self,*args,**kwargs):
        self.proposal=OnlineAreaProposal(*args,**kwargs)
        if not np.array_equal(self.proposal.x_edges,self.tape['x_edges']):
            raise RuntimeError('Online proposal geometry differs from frozen graded101')
        if not np.array_equal(self.proposal.t_edges,self.tape['t_edges']):
            raise RuntimeError('Online time grid differs from frozen graded101')
        return self.proposal

    def points(self,nf,method,cfg,initial_rng,sampler_rng,focus_rng,
               adaptive,model,phase,step,enriched):
        if enriched or nf!=self.nf or method not in ('rad','alhs'):
            raise ValueError('Only non-enriched RAD/ALHS at the prescribed budget are supported')
        if phase=='transition':
            points,nfocus=super().points(nf,method,cfg,initial_rng,sampler_rng,focus_rng,
                                         adaptive,model,phase,step,enriched)
            self.online_points.append(points.copy())
            self.online_pdfs.append(self.tape['generation_pdf'][0].copy())
            self.online_weights.append(self.weights_by_set[0].copy())
            return points,nfocus
        if phase!='joint' or adaptive is not self.proposal:
            raise RuntimeError('Unexpected sampler lifecycle')
        index=adaptive.updates
        if index!=len(self.sequence): raise RuntimeError('Online refresh schedule is inconsistent')
        # The unmodified V235 trainer just called adaptive.update(model).
        # This performs real residual evaluations on the CURRENT learner.
        points=adaptive.sample(nf,sampler_rng,stratified=(method=='alhs'))
        pdf=adaptive.last_generation_pdf.copy()
        weights=inverse_density_weights(pdf,self.area)
        if not np.isfinite(weights).all() or (weights<=0).any():
            raise FloatingPointError('Invalid importance weights')
        # q has a uniform physical floor of at least eta/area under the density law.
        if weights.max() > (1.0/cfg['eta'])*(1+1e-8):
            raise RuntimeError('Importance weights exceed the proposal floor bound')
        self.recent.append((points.copy(),pdf.copy()))
        self.active=(points,weights)
        self.phase=phase;self.set_index=index;self.step=int(step);self.calls=0
        self.tensor_cache=None
        self.sequence.append(dict(set_index=index,phase=phase,joint_epoch=int(step),
            points_sha256=sha_points(points),source_points_verified=False,
            points_origin='current_model_online_proposal',q_min=float(pdf.min()),q_max=float(pdf.max()),
            weight_min=float(weights.min()),weight_max=float(weights.max()),
            weight_mean=float(weights.mean()),weight_ess_proxy=float(weights.sum()**2/np.dot(weights,weights)),
            n_x_ge_0999=int((points[:,0]>=.999).sum())))
        self.online_points.append(points.copy());self.online_pdfs.append(pdf.copy())
        self.online_weights.append(weights.copy())
        self.online_uv.append(adaptive.last_probability_coordinates.copy())
        self.online_cells.append(adaptive.last_generation_cell.copy())
        meta=adaptive.snapshot()
        meta.update(joint_epoch=int(step),update=index,
                    n_x_ge_0999_actual=int((points[:,0]>=.999).sum()),
                    expected_n_x_ge_0999=float(nf*adaptive.band_probability(.999,1.)),
                    proposal_updates_from_current_model=True)
        self.refresh_metadata.append(meta)
        return points,0

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        if len(self.online_points)!=len(self.sequence):
            raise RuntimeError('Missing online point/PDF history')
        (path/'joint_start_control.json').write_text(json.dumps(self.shared_start_audit,indent=2))
        pd.DataFrame(self.logs).to_csv(path/'optimizer_objective_trace.csv',index=False)
        pd.DataFrame(self.gradlogs).to_csv(path/'same_state_gradient_comparison.csv',index=False)
        pd.DataFrame(self.sequence).to_csv(path/'online_points_audit.csv',index=False)
        pd.DataFrame(self.refresh_metadata).to_csv(path/'online_proposal_trace.csv',index=False)
        np.savez_compressed(path/'collocation_generation_weights.npz',
            points=np.stack(self.online_points),generation_pdf=np.stack(self.online_pdfs),
            weights=np.stack(self.online_weights),probability_coordinates=np.stack(self.online_uv),
            generation_cell=np.stack(self.online_cells),lbfgs_points=self.lbfgs_points,
            lbfgs_coefficients=self.lbfgs_weights)
        self.proposal.save_history(path/'online_proposal_history.npz')
        (path/'proposal_grid.json').write_text(json.dumps(dict(
            profile='graded101',x_edges=self.proposal.x_edges.tolist(),
            t_edges=self.proposal.t_edges.tolist(),grid_sha256=self.proposal.grid_sha256,
            generating_pdf='normalized_mass / physical_cell_area, before float32 coordinate rounding',
            online_updates=self.proposal.updates,proposal_evaluations=self.proposal.eval_points),indent=2))
        # Correct legacy descriptive metadata that assumed uniform proposal cells.
        fp=path/'sampler_coverage_by_refresh.csv'
        if fp.exists():
            table=pd.read_csv(fp)
            table['candidate_dx']=np.nan
            table['candidate_dx_min']=float(self.proposal.dx.min())
            table['candidate_dx_max']=float(self.proposal.dx.max())
            table['proposal_profile']='graded101'
            table.to_csv(fp,index=False)
