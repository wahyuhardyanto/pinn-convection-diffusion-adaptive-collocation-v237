#!/usr/bin/env python3
"""Area-correct, piecewise-constant residual proposal for the V2.3.4 ablation.

Only the proposal mesh/physical quadrature changes. The density law, IID versus
probability-space LHS generators, and conditional inverse-CDF order are inherited.
Grading is exactly the geometry tested by audit_resolution_v233.graded_edges.
"""
from __future__ import annotations
import hashlib
import time
from typing import Optional
import numpy as np
import torch
import convection_diffusion_benchmark_v2 as core
from audit_resolution_v233 import graded_edges

PROFILE = 'graded101'
SIDE = 101

class AreaResidualAdaptive2D:
    """Mass P[t,x] includes physical cell area; not equal mass per graded cell.

    Uniform test meshes are supported only for unit/regression testing. Production
    V2.3.4 uses 101 graded spatial cells and 101 uniform temporal cells.
    """
    def __init__(self, side: int, eta: float, c: float, D: float,
                 chunk: int, device: torch.device, *, x_edges=None, t_edges=None):
        if int(side) != side or side < 2 or not 0 < eta < 1 or chunk < 1:
            raise ValueError('Invalid proposal resolution, eta or chunk size')
        self.side=int(side);self.eta=float(eta);self.c=float(c);self.D=float(D)
        self.chunk=int(chunk);self.device=device
        self.x_edges=np.array(graded_edges(side) if x_edges is None else x_edges,dtype=np.float64)
        self.t_edges=np.array(np.linspace(0,1,side+1) if t_edges is None else t_edges,dtype=np.float64)
        for e in (self.x_edges,self.t_edges):
            if len(e)!=side+1 or not np.isfinite(e).all() or e[0]!=0 or e[-1]!=1 or not (np.diff(e)>0).all():
                raise ValueError('Edges must be increasing from 0 to 1, with side cells')
        self.dx=np.diff(self.x_edges);self.dt=np.diff(self.t_edges)
        self.area=np.outer(self.dt,self.dx)
        self.domain_area=float(self.area.sum())
        xc=(self.x_edges[:-1]+self.x_edges[1:])/2
        tc=(self.t_edges[:-1]+self.t_edges[1:])/2
        xx,tt=np.meshgrid(xc,tc,indexing='xy')
        # Keep production residual evaluation float32, exactly as old training.
        self.candidates=np.column_stack((xx.ravel(),tt.ravel())).astype(np.float32)
        self.prob=self.area/self.domain_area
        self.field=np.zeros_like(self.area)
        self.density=np.ones_like(self.area)/self.domain_area
        self.mean_abs_residual=0.0
        self.updates=0;self.eval_points=0;self.eval_seconds=0.0
        self.history=[]
        self.last_probability_coordinates=None
        self.last_points_sha256=None
        self.last_endpoint_counts={}
        self.grid_sha256=hashlib.sha256(self.x_edges.tobytes()+self.t_edges.tobytes()).hexdigest()

    def set_residual_field(self, field):
        f=np.asarray(field,dtype=np.float64)
        if f.shape!=self.area.shape or not np.isfinite(f).all() or (f<0).any():
            raise ValueError('Expected finite absolute residual, shape [t,x]')
        mean=float(np.sum(self.area*f)/self.domain_area)
        q=self.eta+(1-self.eta)*f/(mean+1e-12)
        weighted=q*self.area
        norm=float(weighted.sum())
        if not np.isfinite(norm) or norm<=0:
            raise FloatingPointError('Invalid nonuniform proposal normalization')
        self.field=f.copy();self.mean_abs_residual=mean
        self.prob=weighted/norm
        self.density=self.prob/self.area
        if not np.isclose(self.prob.sum(),1.0,rtol=0,atol=1e-12):
            raise AssertionError('Probability mass does not sum to one')

    def update(self, model):
        start=time.perf_counter();old=model.training
        pieces=[]
        try:
            model.eval()
            with torch.enable_grad():
                for i in range(0,len(self.candidates),self.chunk):
                    xy=torch.from_numpy(self.candidates[i:i+self.chunk]).to(self.device)
                    r=core.pde_residual(model,xy,self.c,self.D)
                    pieces.append(np.abs(r.detach().cpu().numpy().reshape(-1)))
            self.set_residual_field(np.concatenate(pieces).reshape(self.side,self.side))
        finally:
            model.train(old)
        self.updates+=1;self.eval_points+=len(self.candidates)
        self.eval_seconds+=time.perf_counter()-start
        # Arrays retained in RAM; written only AFTER training (no training I/O).
        self.history.append({'update':self.updates,'prob':self.prob.copy(),
                             'absolute_residual':self.field.copy(),
                             'mean_abs_residual':self.mean_abs_residual})

    @staticmethod
    def _invert(u, mass, edges):
        cdf=np.cumsum(mass,dtype=np.float64)
        cdf[-1]=1.0
        idx,frac=core.inv_cdf(np.asarray(u),cdf)
        return idx,edges[idx]+frac*(edges[idx+1]-edges[idx])

    def transform(self, uv):
        uv=np.asarray(uv,dtype=np.float64)
        if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all() or (uv<0).any() or (uv>=1).any():
            raise ValueError('Probability coordinates must have shape (n,2) in [0,1)')
        p_t=self.prob.sum(axis=1)
        it,t=self._invert(uv[:,0],p_t,self.t_edges)
        x=np.empty(len(uv),dtype=np.float64)
        for j in np.unique(it):
            mask=it==j
            _,x[mask]=self._invert(uv[mask,1],self.prob[j]/p_t[j],self.x_edges)
        # Match the old sampler output dtype; do not change training arithmetic.
        return np.column_stack((x,t)).astype(np.float32)

    def sample(self,n: int,rng: np.random.Generator,stratified: bool):
        if int(n)!=n or n<1:raise ValueError('Sample size must be positive')
        uv=core.uniform_lhs(n,rng) if stratified else core.uniform_mc(n,rng)
        pts=self.transform(uv)
        self.last_probability_coordinates=uv.copy()
        self.last_points_sha256=hashlib.sha256(pts.tobytes()).hexdigest()
        self.last_endpoint_counts={'sample_n_x_exact_zero':int((pts[:,0]==0).sum()),'sample_n_x_exact_one':int((pts[:,0]==1).sum())}
        return pts

    def band_probability(self,a: float,b: float):
        overlap=np.maximum(0,np.minimum(self.x_edges[1:],b)-np.maximum(self.x_edges[:-1],a))
        return float(np.sum(self.density*np.outer(self.dt,overlap)))

    def snapshot(self):
        return {
            'proposal_profile':PROFILE,'proposal_grid_sha256':self.grid_sha256,
            'candidate_x_min':float(self.candidates[:,0].min()),
            'candidate_x_max':float(self.candidates[:,0].max()),
            'candidate_dx':np.nan, # no single spatial cell width on a graded mesh
            'candidate_dx_min':float(self.dx.min()),'candidate_dx_max':float(self.dx.max()),
            'candidate_points_per_refresh':len(self.candidates),
            'proposal_mean_abs_residual_area':self.mean_abs_residual,
            'proposal_mass_sum':float(self.prob.sum()),
            'proposal_prob_x_ge_099':self.band_probability(.99,1),
            'proposal_prob_x_ge_0999':self.band_probability(.999,1),
            'proposal_prob_x_ge_09999':self.band_probability(.9999,1),
            'proposal_prob_x_le_0001':self.band_probability(0,.001),
            'proposal_area_weighted':True,
            'sample_points_sha256':self.last_points_sha256,
            **self.last_endpoint_counts,
        }

    def save_history(self,path):
        np.savez_compressed(path,x_edges=self.x_edges,t_edges=self.t_edges,
            cell_area=self.area,candidates_float32=self.candidates,
            probabilities=np.stack([h['prob'] for h in self.history]),
            absolute_residual=np.stack([h['absolute_residual'] for h in self.history]),
            mean_abs_residual_area=np.array([h['mean_abs_residual'] for h in self.history]),
            update_index=np.array([h['update'] for h in self.history]),
            evaluation_seconds=self.eval_seconds,eta=self.eta)
