#!/usr/bin/env python3
"""Local CPU unit tests only; no user experiment or source outputs are modified."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from online_context_v236 import OnlineAreaProposal, OnlineObjectiveContext, ReadOnlyTransition
from nonuniform_proposal_v234 import AreaResidualAdaptive2D
from objective_control_v235 import grouped_union_weights
from shared_transition_v235 import tree_hash, cpu_tree, capture_rng, file_sha
import convection_diffusion_online_v236 as runner
import convection_diffusion_pilot_v233 as base


def verify():
    torch.set_num_threads(1)
    results={};runner.verify_package();results['frozen_sources_and_summary_checksums']=True
    for p in runner.HERE.glob('*.py'):
        ast.parse(p.read_text(),filename=p.name,feature_version=(3,9))
    results['python_39_grammar']=True
    a=OnlineAreaProposal(101,.1,1,.001,512,torch.device('cpu'))
    b=AreaResidualAdaptive2D(101,.1,1,.001,512,torch.device('cpu'))
    for f in (np.zeros_like(a.area),np.ones_like(a.area)*3):
        a.set_residual_field(f)
        np.testing.assert_allclose(a.prob,a.area,atol=1e-15,rtol=1e-12)
        np.testing.assert_allclose(a.density,1,rtol=1e-12)
    results['zero_and_constant_residual_uniform_physical_area']=True
    x=(a.x_edges[:-1]+a.x_edges[1:])/2
    t=(a.t_edges[:-1]+a.t_edges[1:])/2
    f=1+np.exp(-(1-x)[None,:]/.001)*(1+t[:,None])
    a.set_residual_field(f);b.set_residual_field(f)
    uv=np.random.default_rng(42).random((4000,2))
    pts=a.transform(uv);old=b.transform(uv)
    assert np.array_equal(pts,old)
    it,ix=a.last_generation_cell.T
    np.testing.assert_array_equal(a.last_generation_pdf,a.prob[it,ix]/a.area[it,ix])
    assert (1/a.last_generation_pdf).max()<=10*(1+1e-10)
    results['original_inverse_rosenblatt_exact_coordinates']=True
    results['pdf_uses_generating_cell_and_physical_area']=True
    g=1+x[None,:]**2+2*t[:,None]
    np.testing.assert_allclose(np.sum(a.prob*g/a.density),np.sum(a.area*g),rtol=2e-15)
    results['discrete_importance_identity']=True
    a.sample(500,np.random.default_rng(89),True)
    uv=a.last_probability_coordinates
    for k in range(2):assert len(np.unique(np.floor(uv[:,k]*500).astype(int)))==500
    results['probability_space_lhs_strata']=True
    # Uniform-generator stream consumed identically by old and new sampler.
    for stratified in (False,True):
        g1=np.random.default_rng(57);g2=np.random.default_rng(57)
        p1=a.sample(31,g1,stratified);p2=b.sample(31,g2,stratified)
        assert np.array_equal(p1,p2)
        assert tree_hash(g1.bit_generator.state)==tree_hash(g2.bit_generator.state)
    results['same_rng_consumption_as_old_sampling']=True
    u=np.array([[.2,.1],[.2,.1],[.8,.7]],np.float32)
    sets=[u[:2],u[2:]];pdfs=[np.array([.5,2]),np.array([4.])]
    unique,coeff,duplicates=grouped_union_weights(sets,pdfs)
    vals=np.sum(unique,axis=1)**2
    direct=np.mean(np.sum(u,axis=1)**2/np.concatenate(pdfs))
    np.testing.assert_allclose(np.mean(coeff*vals),direct,rtol=1e-7)
    assert duplicates==1
    results['lbfgs_per_generation_pdf_duplicate_sums']=True
    cfg=runner.configuration('smoke')
    model=base.HardBothBoundaryLayerPINN(cfg['layers'],.001)
    before=tree_hash(model.state_dict());rng=torch.get_rng_state().clone()
    for p in model.parameters():p.grad=torch.ones_like(p)
    grads=[p.grad.clone() for p in model.parameters()]
    a.update(model)
    assert a.eval_points==10201 and a.updates==1
    assert tree_hash(model.state_dict())==before and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(g,p.grad) for g,p in zip(grads,model.parameters()))
    history=a.history[0]['prob'].copy()
    with torch.no_grad():next(model.parameters()).add_(.03)
    a.update(model)
    assert a.eval_points==20402
    assert np.array_equal(history,a.history[0]['prob'])
    assert not np.array_equal(a.history[0]['prob'],a.history[1]['prob'])
    results['online_update_evaluates_current_model_without_rng_or_parameter_update']=True
    results['proposal_history_is_immutable_per_refresh']=True
    # Pre-rounding generation cells are explicit, including rounded x=1.
    a.set_residual_field(np.ones_like(a.area))
    a.transform(np.array([[.5,np.nextafter(1.,0.)]]))
    assert a.last_generation_cell[0,1]==100
    assert a.last_pre_rounding_points[0,0]<=1
    results['edge_cell_rounding_pdf_is_preserved']=True
    with tempfile.TemporaryDirectory() as td:
        folder=Path(td)
        m=torch.nn.Linear(2,1);opt=torch.optim.Adam(m.parameters(),lr=.001)
        xx=torch.tensor([[.2,.8]]);(m(xx)**2).sum().backward();opt.step()
        warm=cpu_tree(m.state_dict());generators={'g':np.random.default_rng(4)}
        initial=base.v21.state_hash(warm)
        bundle=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),
                    rng=cpu_tree(capture_rng(torch.device('cpu'),generators)),warm_state=warm,
                    initial_state_sha256=initial,transition_points_sha256='fixture',
                    history=[],objective_logs=[],gradient_logs=[],source_warmup_seconds=0.,
                    source_transition_seconds=0.,creator_objective='unit_test_fixture')
        torch.save(bundle,folder/'common_transition.pt')
        meta=dict(signature={'unit_test':True},file_sha256=file_sha(folder/'common_transition.pt'),
                  content_hashes={k:tree_hash(bundle[k]) for k in ('model','optimizer','rng','warm_state')})
        (folder/'manifest.json').write_text(json.dumps(meta))
        hashes=(file_sha(folder/'common_transition.pt'),file_sha(folder/'manifest.json'))
        locked=ReadOnlyTransition(folder,torch.device('cpu'),meta['file_sha256'])
        locked.begin(m,opt,initial)
        with torch.no_grad():next(m.parameters()).add_(1.)
        torch.manual_seed(238);generators['g'].random(10)
        audit=locked.finish(m,opt,generators,warm,[],[],[],'fixture',0.,0.)
        assert all(audit['exact_checks'].values())
        assert hashes==(file_sha(folder/'common_transition.pt'),file_sha(folder/'manifest.json'))
        bad=False
        try:ReadOnlyTransition(folder,torch.device('cpu'),'0'*64)
        except RuntimeError:bad=True
        assert bad
    results['exact_readonly_model_adam_rng_restore']=True
    results['source_checksum_mismatch_rejected']=True
    return {'tests':results,'all_passed':all(results.values()),
            'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
            'no_mps_test':True,'training_accuracy_claimed':False}

if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
