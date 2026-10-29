#!/usr/bin/env python3
"""V2.3.7 unified four-sampler training context.

All four samplers optimize the same physical-uniform residual objective.
MC-R/LHS-R have q=1 on the unit domain. RAD-IID/ALHS-R use the same graded101
current-model proposal and ordinary importance correction 1/q.
The transition design is method-independent uniform LHS so one exact prefix can
be shared across all four methods for a given (budget, seed).
"""
from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import convection_diffusion_pilot_v233 as base
from objective_control_v235 import inverse_density_weights, grouped_union_weights, sha_points
from online_context_v236 import OnlineAreaProposal

METHODS=('mc','lhs','rad','alhs')

class UnifiedObjectiveContext:
    objective='importance_uniform'
    def __init__(self,cfg,gradient_audit=True):
        self.cfg=cfg; self.gradient_audit=gradient_audit
        self.area=1.0; self.nf=int(cfg['n_collocation'][0])
        self.sequence=[]; self.logs=[]; self.gradlogs=[]
        self.recent=deque(maxlen=int(cfg['lbfgs_union_sets']))
        self.active=None; self.tensor_cache=None; self.phase='uninitialized'; self.calls=0
        self.transition_hash=None; self.shared_transition=None; self.shared_start_audit={}
        self.proposal=None; self.lbfgs_duplicate_count=None; self.lbfgs_weights=None; self.lbfgs_points=None
        self.point_history=[]; self.pdf_history=[]; self.weight_history=[]; self.refresh_metadata=[]

    def make_proposal(self,*args,**kwargs):
        self.proposal=OnlineAreaProposal(*args,**kwargs)
        return self.proposal

    def _activate(self,points,pdf,phase,step,origin):
        points=np.asarray(points,dtype=np.float32)
        pdf=np.asarray(pdf,dtype=np.float64).reshape(-1)
        if points.shape!=(self.nf,2) or pdf.shape!=(self.nf,): raise ValueError('Point/PDF shape mismatch')
        weights=inverse_density_weights(pdf,self.area)
        self.recent.append((points.copy(),pdf.copy()))
        self.active=(points.copy(),weights.copy()); self.tensor_cache=None
        self.phase=phase; self.step=int(step); self.calls=0
        self.point_history.append(points.copy()); self.pdf_history.append(pdf.copy()); self.weight_history.append(weights.copy())
        self.sequence.append(dict(set_index=len(self.sequence),phase=phase,joint_epoch=int(step),
            points_sha256=sha_points(points),points_origin=origin,q_min=float(pdf.min()),q_max=float(pdf.max()),
            weight_min=float(weights.min()),weight_max=float(weights.max()),weight_mean=float(weights.mean()),
            weight_ess_proxy=float(weights.sum()**2/np.dot(weights,weights)),n_x_ge_0999=int((points[:,0]>=.999).sum())))
        return points,0

    def points(self,nf,method,cfg,initial_rng,sampler_rng,focus_rng,adaptive,model,phase,step,enriched):
        if enriched or nf!=self.nf or method not in METHODS: raise ValueError('Unexpected sampler/budget/enrichment')
        if phase=='transition':
            # Method-independent common transition for exact four-way branching.
            pts=base.core.uniform_lhs(nf,initial_rng).astype(np.float32)
            return self._activate(pts,np.ones(nf,dtype=np.float64),'transition',step,'shared_uniform_lhs_transition')
        if phase!='joint': raise RuntimeError('Unexpected sampler lifecycle')
        if method=='mc':
            pts=base.core.uniform_mc(nf,sampler_rng).astype(np.float32); pdf=np.ones(nf,dtype=np.float64)
            out=self._activate(pts,pdf,phase,step,'uniform_iid')
            self.refresh_metadata.append(dict(joint_epoch=int(step),method=method,proposal='uniform',
                n_x_ge_0999_actual=int((pts[:,0]>=.999).sum()),expected_n_x_ge_0999=float(nf*0.001),proposal_evaluations=0))
            return out
        if method=='lhs':
            pts=base.core.uniform_lhs(nf,sampler_rng).astype(np.float32); pdf=np.ones(nf,dtype=np.float64)
            out=self._activate(pts,pdf,phase,step,'uniform_probability_lhs')
            self.refresh_metadata.append(dict(joint_epoch=int(step),method=method,proposal='uniform',
                n_x_ge_0999_actual=int((pts[:,0]>=.999).sum()),expected_n_x_ge_0999=float(nf*0.001),proposal_evaluations=0))
            return out
        if adaptive is not self.proposal: raise RuntimeError('Adaptive proposal lifecycle mismatch')
        pts=adaptive.sample(nf,sampler_rng,stratified=(method=='alhs'))
        pdf=adaptive.last_generation_pdf.copy()
        if not np.isfinite(pdf).all() or (pdf<=0).any(): raise FloatingPointError('Invalid generating PDF')
        if inverse_density_weights(pdf,self.area).max()>(1.0/cfg['eta'])*(1+1e-8):
            raise RuntimeError('Importance weights exceed proposal floor bound')
        out=self._activate(pts,pdf,phase,step,'current_model_'+('lhs' if method=='alhs' else 'iid'))
        meta=adaptive.snapshot(); meta.update(joint_epoch=int(step),method=method,
            n_x_ge_0999_actual=int((pts[:,0]>=.999).sum()),expected_n_x_ge_0999=float(nf*adaptive.band_probability(.999,1.)),
            proposal_evaluations=int(adaptive.eval_points))
        self.refresh_metadata.append(meta)
        return out

    def after_transition(self,model):
        state_cpu={name:t.detach().cpu() for name,t in model.state_dict().items()}
        self.transition_hash=base.v21.state_hash(state_cpu)

    def prepare_lbfgs(self,union):
        points,pdfs=zip(*self.recent)
        unique,weights,dup=grouped_union_weights(list(points),list(pdfs),self.area)
        if not np.array_equal(union,unique): raise RuntimeError('L-BFGS union alignment failed')
        self.active=(unique,weights); self.phase='lbfgs'; self.calls=0; self.tensor_cache=None
        self.lbfgs_duplicate_count=dup; self.lbfgs_weights=weights; self.lbfgs_points=unique

    def _weights(self,res_xt):
        if self.active is None: raise RuntimeError('Loss requested before collocation')
        if self.tensor_cache is not None and self.tensor_cache[0] is res_xt: return self.tensor_cache[1]
        points,coeff=self.active; actual=res_xt.detach().cpu().numpy()
        if not np.array_equal(actual,points): raise RuntimeError('Training weights applied to unknown points')
        w=torch.as_tensor(coeff,dtype=res_xt.dtype,device=res_xt.device).reshape(-1,1).detach()
        self.tensor_cache=(res_xt,w); return w

    def full_losses(self,model,res_xt,ic,ic_y,bc,cfg,res_weight):
        w=self._weights(res_xt)
        res=base.core.pde_residual(model,res_xt,cfg['c'],cfg['D']); r2=res.square()
        ordinary=r2.mean(); corrected=(w*r2).mean()
        # transition q=1, therefore corrected==ordinary and one objective applies to every branch
        pde=corrected
        icloss,bcloss=base.core.constraint_terms(model,ic,ic_y,bc)
        total=res_weight*pde+cfg['w_ic']*icloss+cfg['w_bc']*bcloss
        self.calls+=1
        if self.calls==1 or self.calls%100==0:
            wd=w.detach().cpu().numpy().ravel(); right=(res_xt[:,0:1]>=.999).to(res.dtype)
            self.logs.append(dict(phase=self.phase,set_index=len(self.sequence)-1,call_in_set=self.calls,
                objective='importance_uniform',pde_weight=float(res_weight),optimizer_loss=float(total.detach()),
                ordinary_pde=float(ordinary.detach()),is_uniform_estimate=float(corrected.detach()),
                ic_loss=float(icloss.detach()),bc_loss=float(bcloss.detach()),weight_min=float(wd.min()),
                weight_max=float(wd.max()),weight_mean=float(wd.mean()),weight_ess_proxy=float(wd.sum()**2/np.dot(wd,wd)),
                right_strip_ordinary_contribution=float((right*r2).mean().detach()),
                right_strip_is_contribution=float((right*w*r2).mean().detach())))
        if self.gradient_audit and self.calls==1:
            params=tuple(model.parameters())
            ga=torch.autograd.grad(ordinary,params,retain_graph=True,allow_unused=True)
            gb=torch.autograd.grad(corrected,params,retain_graph=True,allow_unused=True)
            a=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,ga)])
            b=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gb)])
            an=torch.linalg.vector_norm(a); bn=torch.linalg.vector_norm(b)
            cos=float((a@b/(an*bn).clamp_min(1e-30)).detach())
            self.gradlogs.append(dict(phase=self.phase,set_index=len(self.sequence)-1,same_model_same_points=True,
                pde_gradient_norm_plain=float(an.detach()),pde_gradient_norm_is=float(bn.detach()),
                cosine_plain_vs_is=cos,pde_weight=float(res_weight),gradients_written_to_optimizer=False))
        return total,pde,icloss,bcloss

    def save(self,path):
        path=Path(path); path.mkdir(parents=True,exist_ok=True)
        (path/'joint_start_control.json').write_text(json.dumps(self.shared_start_audit,indent=2),encoding='utf-8')
        pd.DataFrame(self.logs).to_csv(path/'optimizer_objective_trace.csv',index=False)
        pd.DataFrame(self.gradlogs).to_csv(path/'same_state_gradient_comparison.csv',index=False)
        pd.DataFrame(self.sequence).to_csv(path/'sampling_points_audit.csv',index=False)
        pd.DataFrame(self.refresh_metadata).to_csv(path/'proposal_trace.csv',index=False)
        np.savez_compressed(path/'collocation_generation_weights.npz',points=np.stack(self.point_history),
            generation_pdf=np.stack(self.pdf_history),weights=np.stack(self.weight_history),
            lbfgs_points=self.lbfgs_points,lbfgs_coefficients=self.lbfgs_weights)
        if self.proposal is not None:
            self.proposal.save_history(path/'online_proposal_history.npz')
            (path/'proposal_grid.json').write_text(json.dumps(dict(profile='graded101',
                x_edges=self.proposal.x_edges.tolist(),t_edges=self.proposal.t_edges.tolist(),
                grid_sha256=self.proposal.grid_sha256,online_updates=self.proposal.updates,
                proposal_evaluations=self.proposal.eval_points),indent=2),encoding='utf-8')
