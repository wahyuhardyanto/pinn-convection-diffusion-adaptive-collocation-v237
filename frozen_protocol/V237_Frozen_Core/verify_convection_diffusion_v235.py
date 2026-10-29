#!/usr/bin/env python3
"""Fast mathematical/provenance tests; --integration also trains CPU smoke controls.
No user checkpoint, baseline or output is overwritten.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, tempfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from objective_control_v235 import inverse_density_weights,grouped_union_weights,ObjectiveContext
import convection_diffusion_objective_v235 as run
from nonuniform_proposal_v234 import AreaResidualAdaptive2D


def tests():
    prov=run.verify_sources();result={'status':'passed','mathematical_checks':{}}
    checks=result['mathematical_checks']
    sampler=AreaResidualAdaptive2D(101,.1,1.,.001,512,torch.device('cpu'))
    for label,value in [('zero',0.),('constant',4.)]:
        sampler.set_residual_field(np.full(sampler.area.shape,value))
        assert np.allclose(sampler.density,1.,rtol=0,atol=1e-12)
        assert np.allclose(inverse_density_weights(sampler.density),1.,rtol=0,atol=1e-12)
        checks[label+'_residual_uniform_area_weights']=True
    xx=(sampler.x_edges[:-1]+sampler.x_edges[1:])/2
    tt=(sampler.t_edges[:-1]+sampler.t_edges[1:])/2
    x,t=np.meshgrid(xx,tt,indexing='xy')
    field=.01+np.exp(-(1-x)/.001)*(1+t)
    sampler.set_residual_field(field)
    f=(1+x+2*t)**2
    corrected=float(np.sum(sampler.prob*f/sampler.density))
    exact=float(np.sum(sampler.area*f))
    uncorrected=float(np.sum(sampler.prob*f))
    assert abs(corrected-exact)<1e-12
    assert abs(uncorrected-exact)>1e-3
    assert np.max(1/sampler.density)<=10.000000001
    checks.update(cell_quadrature_importance_identity_error=abs(corrected-exact),
        ordinary_distribution_shift_detected=True,unit_domain_eta01_weights_bounded_by10=True)
    theta=torch.tensor(1.3,dtype=torch.float64,requires_grad=True)
    values=torch.from_numpy((x+2*t).reshape(-1,1))
    p=torch.from_numpy(sampler.prob.reshape(-1,1))
    q=torch.from_numpy(sampler.density.reshape(-1,1)).detach()
    a=torch.from_numpy(sampler.area.reshape(-1,1))
    # Exact expectation over the cell-centered finite test problem.
    corrected_obj=(p*(theta*values)**2/q).sum()
    target=(a*(theta*values)**2).sum()
    ga=torch.autograd.grad(corrected_obj,theta,retain_graph=True)[0]
    gb=torch.autograd.grad(target,theta)[0]
    assert torch.allclose(ga,gb,rtol=0,atol=1e-12)
    assert not q.requires_grad
    checks['fixed_integrand_gradient_identity']=True
    pts1=np.array([[.2,.3],[.4,.5]],dtype=np.float32)
    pts2=np.array([[.2,.3],[.8,.9]],dtype=np.float32)
    qs=[np.array([.4,2.]),np.array([1.2,.6])]
    unique,coeff,ndup=grouped_union_weights([pts1,pts2],qs)
    f_all=np.sum(np.concatenate([pts1,pts2]).astype('float64'),axis=1)**2
    direct=np.mean(f_all*np.concatenate([1/qs[0],1/qs[1]]))
    grouped=np.mean(np.sum(unique.astype('float64'),axis=1)**2*coeff)
    assert ndup==1 and abs(direct-grouped)<1e-12
    checks['lbfgs_generation_density_and_duplicate_coefficients']=True
    count=0;normalization_max=0.;density_max=0.
    for row in prov['records']:
        with np.load(run.HERE/row['tape_path'],allow_pickle=False) as z:
            area=np.outer(np.diff(z['t_edges']),np.diff(z['x_edges']))
            for k,pts in enumerate(z['points']):
                assert hashlib.sha256(pts.tobytes()).hexdigest()==str(z['point_sha256'][k]);count+=1
                if k:
                    p=z['probabilities'][k-1]
                    normalization_max=max(normalization_max,abs(float(p.sum())-1))
                    it,ix=z['generation_t_cell'][k],z['generation_x_cell'][k]
                    pdf=p[it,ix]/area[it,ix]
                    density_max=max(density_max,float(np.max(abs(pdf-z['generation_pdf'][k]))))
            assert np.all(z['generation_pdf']>0)
    assert normalization_max<1e-12 and density_max<1e-12
    checks.update(point_sets_checked=count,stored_pdf_matches_generating_cell=True,
                  max_probability_normalization_error=normalization_max)
    # Syntax intended for user's Python 3.9; runtime tested separately.
    for p in run.HERE.glob('*.py'):
        ast.parse(p.read_text(),filename=p.name,feature_version=(3,9))
    checks['python39_syntax']=True
    result['runtime']={'python':__import__('sys').version,'torch':torch.__version__,'numpy':np.__version__}
    return result


@contextmanager
def replay_original(ctx):
    base=run.base
    old=(base.method_points,base.core.ResidualAdaptive2D)
    base.method_points=ctx.points
    base.core.ResidualAdaptive2D=ctx.make_proposal
    try:yield
    finally:base.method_points,base.core.ResidualAdaptive2D=old


def integration():
    """Old trainer+plain replay versus new trainer+plain loss; CPU bitwise check."""
    torch.set_num_threads(1)
    cfg,_,_=run.configuration('smoke');ref,sg=run.references(cfg)
    dev=torch.device('cpu');out=[]
    for method in ('rad','alhs'):
        with tempfile.TemporaryDirectory(prefix='v235_test_') as temp:
            t=Path(temp);tape=run.HERE/'replay_tapes'/'smoke'/f'{method}_seed000.npz'
            oldctx=ObjectiveContext(tape,'plain',cfg,gradient_audit=False)
            with replay_original(oldctx):
                a=run.base.run_candidate(run.base.CANDIDATE_VARIANT,method,cfg,ref,sg,dev,t/'old','bc_feasible',seed=0)
            ctx=ObjectiveContext(tape,'plain',cfg,gradient_audit=True)
            b=run.run_objective_candidate(run.base.CANDIDATE_VARIANT,method,cfg,ref,sg,dev,t/'new','bc_feasible',seed=0,objective_context=ctx)
            sa=torch.load(t/'old/final_model_state.pt',map_location='cpu',weights_only=True)
            sb=torch.load(t/'new/final_model_state.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(sa[k],sb[k]) for k in sa)
            assert a['selected_checkpoint']==b['selected_checkpoint']
            assert a['sampling_sha256']==b['sampling_sha256']
            for metric in ('space_time_rel_l2','final_rel_l2','dense_physics_loss'):
                assert a[metric]==b[metric],(metric,a[metric],b[metric])
            out.append({'method':method,'plain_replay_old_vs_new_weights_bitwise_equal':True,
                        'diagnostic_gradients_did_not_change_optimizer':True})
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration',action='store_true')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    result=tests()
    if args.integration:result['integration']=integration()
    print(json.dumps(result,indent=2))
    if args.output:args.output.write_text(json.dumps(result,indent=2))
