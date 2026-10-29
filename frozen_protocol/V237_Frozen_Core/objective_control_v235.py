#!/usr/bin/env python3
"""Fixed-point replay and ordinary-IS training loss; no importance self-normalization.

Target = mean physical uniform squared residual. Domain area is 1 for this study.
PDF = generating normalized cell mass / physical cell area (not cell mass alone).
Historical float32 coordinates have been verified against source point-set hashes.
The nominal density is retained from the generating pre-rounding cell.
"""
from __future__ import annotations
from collections import deque
import hashlib
import numpy as np
import torch
import convection_diffusion_pilot_v233 as base


def sha_points(points):
    return hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest()


def inverse_density_weights(pdf, domain_area=1.0):
    pdf = np.asarray(pdf, dtype=np.float64).reshape(-1)
    if not np.isfinite(pdf).all() or (pdf <= 0).any() or domain_area <= 0:
        raise ValueError('A positive, finite GENERATING probability density is required')
    return 1.0 / (float(domain_area) * pdf)


def grouped_union_weights(point_sets, pdf_sets, domain_area=1.0):
    """Evaluate sum(r_i^2/(area*q_gen_i))/total draws on unique coordinates.

    Original occurrence weights are SUMMED, not averaged over duplicated points.
    The returned coefficient is suitable for torch.mean(coeff * residual**2).
    No mixture denominator depending on future adaptive proposals is introduced.
    """
    if len(point_sets) != len(pdf_sets) or not point_sets:
        raise ValueError('Point sets and density sets must align')
    points = np.concatenate(point_sets).astype(np.float32)
    w = np.concatenate([inverse_density_weights(q, domain_area) for q in pdf_sets])
    if len(points) != len(w): raise ValueError('Point/PDF shape mismatch')
    unique, inv = np.unique(points, axis=0, return_inverse=True)
    coeff = np.bincount(inv, weights=w, minlength=len(unique)) * len(unique) / len(points)
    return unique, coeff, len(points) - len(unique)


class ReplayDensity:
    """Read-only donor probability sequence, independent of the current learner."""
    def __init__(self, tape):
        self.tape=tape; self.side=len(tape['x_edges'])-1
        self.x_edges=tape['x_edges'];self.t_edges=tape['t_edges']
        self.area=np.outer(np.diff(self.t_edges),np.diff(self.x_edges))
        self.prob=self.area/self.area.sum()
        xc=(self.x_edges[:-1]+self.x_edges[1:])/2
        tc=(self.t_edges[:-1]+self.t_edges[1:])/2
        x,t=np.meshgrid(xc,tc,indexing='xy')
        self.candidates=np.column_stack([x.ravel(),t.ravel()]).astype(np.float32)
        self.updates=0;self.eval_points=0;self.eval_seconds=0.0
    def update(self, model):
        # Intentionally DO NOT evaluate the learner or change its parameters.
        if self.updates>=len(self.tape['probabilities']):
            raise RuntimeError('Replay exhausted: schedule differs from donor')
        self.prob=self.tape['probabilities'][self.updates].copy()
        self.updates+=1


class ObjectiveContext:
    """Training-only loss hook. Validation never calls this object."""
    def __init__(self, tape_file, objective, cfg, gradient_audit=True):
        if objective not in ('plain','importance_uniform'):
            raise ValueError('Unknown objective')
        with np.load(tape_file,allow_pickle=False) as z:
            self.tape={k:z[k].copy() for k in z.files}
        self.objective=objective;self.cfg=cfg;self.gradient_audit=gradient_audit
        self.area=float((self.tape['x_edges'][-1]-self.tape['x_edges'][0])*
                        (self.tape['t_edges'][-1]-self.tape['t_edges'][0]))
        if self.area!=1.0:raise ValueError('Frozen study requires the unit domain')
        self.nf=int(cfg['n_collocation'][0]);self.sequence=[];self.logs=[];self.gradlogs=[]
        self.recent=deque(maxlen=cfg['lbfgs_union_sets'])
        self.active=None;self.tensor_cache=None;self.phase='uninitialized';self.calls=0
        self.transition_hash=None;self.generation_pdf_detached=True
        expected=1+(cfg['joint_epochs']-1)//cfg['resample_every']+1
        if self.tape['points'].shape!=(expected,self.nf,2):
            raise ValueError('Replay tape shape differs from training schedule')
        if self.tape['generation_pdf'].shape!=(expected,self.nf):
            raise ValueError('Replay PDF metadata malformed')
        for i,points in enumerate(self.tape['points']):
            if sha_points(points)!=str(self.tape['point_sha256'][i]):
                raise RuntimeError('Replay points changed')
        self.weights_by_set=inverse_density_weights(self.tape['generation_pdf'],self.area).reshape(expected,self.nf)
        self.lbfgs_duplicate_count=None
        self.shared_transition=None
        self.shared_start_audit={}

    def make_proposal(self,*args,**kwargs):
        self.proposal=ReplayDensity(self.tape)
        return self.proposal

    def points(self,nf,method,cfg,initial_rng,sampler_rng,focus_rng,
               adaptive,model,phase,step,enriched):
        if enriched or nf!=self.nf:raise ValueError('Unexpected enrichment or budget')
        index=0 if phase=='transition' else adaptive.updates
        if index!=len(self.sequence):raise RuntimeError('Replay order mismatch')
        points=self.tape['points'][index].copy()
        pdf=self.tape['generation_pdf'][index].copy()
        self.recent.append((points,pdf))
        self.active=(points,self.weights_by_set[index].copy())
        self.phase=phase;self.set_index=index;self.step=int(step);self.calls=0
        self.tensor_cache=None
        self.sequence.append({'set_index':index,'phase':phase,'joint_epoch':int(step),
            'points_sha256':sha_points(points),'source_points_verified':True,
            'q_min':float(pdf.min()),'q_max':float(pdf.max()),
            'weight_min':float(self.weights_by_set[index].min()),
            'weight_max':float(self.weights_by_set[index].max()),
            'weight_mean':float(self.weights_by_set[index].mean()),
            'n_x_ge_0999':int((points[:,0]>=.999).sum())})
        return points,0

    def after_transition(self, model):
        """Hash CPU copies only; do not move the live model or call observers."""
        state_cpu = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }
        self.transition_hash = base.v21.state_hash(state_cpu)

    def prepare_lbfgs(self,union):
        points,pdfs=zip(*self.recent)
        unique,weights,dup=grouped_union_weights(list(points),list(pdfs),self.area)
        if not np.array_equal(union,unique):raise RuntimeError('L-BFGS union alignment failed')
        self.active=(unique,weights);self.phase='lbfgs';self.calls=0;self.tensor_cache=None
        self.lbfgs_duplicate_count=dup
        self.lbfgs_weights=weights;self.lbfgs_points=unique

    def _weights(self,res_xt):
        if self.active is None:raise RuntimeError('Loss requested before collocation')
        if self.tensor_cache is not None and self.tensor_cache[0] is res_xt:
            return self.tensor_cache[1]
        points,coeff=self.active
        actual=res_xt.detach().cpu().numpy()
        if not np.array_equal(actual,points):
            raise RuntimeError('Refusing to apply training weights to unknown/validation points')
        w=torch.as_tensor(coeff,dtype=res_xt.dtype,device=res_xt.device).reshape(-1,1).detach()
        if w.shape!=(len(res_xt),1) or w.requires_grad:raise AssertionError('Unsafe weight broadcasting/gradient')
        self.tensor_cache=(res_xt,w)
        return w

    def full_losses(self,model,res_xt,ic,ic_y,bc,cfg,res_weight):
        w=self._weights(res_xt)
        res=base.core.pde_residual(model,res_xt,cfg['c'],cfg['D'])
        if res.shape!=w.shape:raise ValueError('Residual/weight dimensions must be (n,1)')
        r2=res.square()
        ordinary=r2.mean()
        corrected=(w*r2).mean() # ordinary IS, NOT divided by sum(w)
        if self.phase=='transition':
            if not np.array_equal(self.weights_by_set[0], np.ones(self.nf)):
                raise RuntimeError('Common transition requires exactly uniform generating density')
            pde=ordinary  # ONE autograd path for both objectives before branching
        else:
            pde=ordinary if self.objective=='plain' else corrected
        icloss,bcloss=base.core.constraint_terms(model,ic,ic_y,bc)
        total=res_weight*pde+cfg['w_ic']*icloss+cfg['w_bc']*bcloss
        self.calls+=1
        log=self.calls==1 or self.calls%100==0
        if log:
            wd=w.detach().cpu().numpy().ravel()
            right=(res_xt[:,0:1]>=.999).to(res.dtype)
            self.logs.append(dict(phase=self.phase,set_index=self.set_index,call_in_set=self.calls,
                objective=self.objective,pde_weight=float(res_weight),
                optimizer_loss=float(total.detach()),ordinary_pde=float(ordinary.detach()),
                is_uniform_estimate=float(corrected.detach()),ic_loss=float(icloss.detach()),bc_loss=float(bcloss.detach()),
                weight_min=float(wd.min()),weight_max=float(wd.max()),weight_mean=float(wd.mean()),
                weight_ess_proxy=float(wd.sum()**2/np.dot(wd,wd)),
                right_strip_ordinary_contribution=float((right*r2).mean().detach()),
                right_strip_is_contribution=float((right*w*r2).mean().detach())))
        if self.gradient_audit and self.calls==1:
            params=tuple(model.parameters())
            ga=torch.autograd.grad(ordinary,params,retain_graph=True,allow_unused=True)
            gb=torch.autograd.grad(corrected,params,retain_graph=True,allow_unused=True)
            a=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,ga)])
            b=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gb)])
            an=torch.linalg.vector_norm(a);bn=torch.linalg.vector_norm(b)
            cosine=float((a@b/(an*bn).clamp_min(1e-30)).detach())
            self.gradlogs.append(dict(phase=self.phase,set_index=self.set_index,
                same_model_same_points=True,pde_gradient_norm_plain=float(an.detach()),
                pde_gradient_norm_is=float(bn.detach()),cosine_plain_vs_is=cosine,
                pde_weight=float(res_weight),gradients_written_to_optimizer=False))
        return total,pde,icloss,bcloss

    def save(self,path):
        import pandas as pd
        path.mkdir(parents=True,exist_ok=True)
        if self.shared_start_audit:
            import json
            (path/'joint_start_control.json').write_text(
                json.dumps(self.shared_start_audit,indent=2),encoding='utf-8')
        pd.DataFrame(self.logs).to_csv(path/'optimizer_objective_trace.csv',index=False)
        pd.DataFrame(self.gradlogs).to_csv(path/'same_state_gradient_comparison.csv',index=False)
        pd.DataFrame(self.sequence).to_csv(path/'replay_points_audit.csv',index=False)
        np.savez_compressed(path/'collocation_generation_weights.npz',
            points=self.tape['points'],generation_pdf=self.tape['generation_pdf'],
            weights=self.weights_by_set,lbfgs_points=self.lbfgs_points,
            lbfgs_coefficients=self.lbfgs_weights)
