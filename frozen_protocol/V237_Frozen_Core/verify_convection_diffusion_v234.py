#!/usr/bin/env python3
"""No-install verification for nonuniform proposal and frozen training protocol.
--integration additionally runs TWO short CPU control trainings; not publication data.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,tempfile
from pathlib import Path
import numpy as np
import torch
import convection_diffusion_proposal_v234 as app
from nonuniform_proposal_v234 import AreaResidualAdaptive2D


def run_checks():
    results={}
    baseline,manifest,prov=app.load_frozen()
    cfg=app.config('ablation')
    difference={k for k in cfg if cfg[k]!=manifest['pde_config'][k]}
    assert difference=={'adapt_grid_side'}
    assert cfg['n_collocation']==[500] and cfg['val_grid_side']==31
    results['only_config_change_is_adapt_grid_side']=True
    results['frozen_20_baseline_and_source_hashes_match']=True
    cpu=torch.device('cpu');obj=AreaResidualAdaptive2D(101,.1,1,.001,512,cpu)
    assert np.array_equal(obj.x_edges,app.audit.graded_edges(101))
    assert len(obj.candidates)==10201 and abs(obj.area.sum()-1)<1e-14
    results['graded101_geometry_matches_resolution_audit']=True
    for val in (0.,1.,3.2):
        obj.set_residual_field(np.full((101,101),val))
        assert np.allclose(obj.prob,obj.area,rtol=2e-14,atol=1e-18)
        assert abs(obj.band_probability(.999,1)-.001)<1e-13
        uv=np.random.default_rng(52).random((500,2));xt=obj.transform(uv)
        assert np.max(np.abs(xt-uv[:,::-1]))<3e-8
    results['zero_and_constant_residual_give_physical_uniform']=True
    # Same density and same random generator on a uniform mesh reduce to old code.
    side=51;e=np.linspace(0,1,side+1)
    test=AreaResidualAdaptive2D(side,.1,1,.001,512,cpu,x_edges=e,t_edges=e)
    old=app.core.ResidualAdaptive2D(side,.1,1,.001,512,cpu)
    f=np.random.default_rng(4).random((side,side))+1e-4
    test.set_residual_field(f);q=.1+.9*f/(f.mean()+1e-12);old.prob=q/q.sum()
    assert np.max(np.abs(test.prob-old.prob))<1e-16
    for stratified in (False,True):
        ra=np.random.default_rng(17);rb=np.random.default_rng(17)
        a=test.sample(500,ra,stratified);b=old.sample(500,rb,stratified)
        assert np.array_equal(a,b),(stratified,np.max(abs(a-b)))
        assert ra.bit_generator.state==rb.bit_generator.state
    results['uniform_mesh_matches_old_sample_values_and_rng_exactly']=True
    # Geometric finite-volume approximation of a narrow exponential residual.
    x=(obj.x_edges[:-1]+obj.x_edges[1:])/2
    f=np.broadcast_to(np.exp((x-1)/.001),(101,101)).copy()
    obj.set_residual_field(f)
    approx=obj.band_probability(.999,1)
    exact=.1*.001+.9*(1-np.exp(-1))/(1-np.exp(-1000))
    assert abs(approx/exact-1)<.03
    results['exponential_boundary_mass_relative_discretization_error']=abs(approx/exact-1)
    # Count observed band samples versus exact piecewise-constant probability.
    rr=np.random.default_rng(54);s=obj.sample(100000,rr,False)
    freq=float((s[:,0]>=.999).mean())
    assert abs(freq-approx)<.007
    results['iid_band_mass_empirical']=freq
    results['iid_band_mass_expected_piecewise']=approx
    obj.sample(500,np.random.default_rng(78),True)
    assert all(len(np.unique(np.floor(obj.last_probability_coordinates[:,k]*500)))==500 for k in range(2))
    results['alhs_preserves_all_500_probability_strata_per_axis']=True
    # Invalid states fail clearly, without being silently normalized.
    for field in (np.full((101,101),np.nan),-np.ones((101,101))):
        try:obj.set_residual_field(field)
        except ValueError:pass
        else:raise AssertionError('Invalid field accepted')
    results['invalid_fields_rejected']=True
    torch.manual_seed(0)
    model=app.base.HardBothBoundaryLayerPINN(cfg['layers'],.001).to(cpu)
    state=app.base.v21.model_checkpoint(model);h=app.base.v21.state_hash(state)
    rng_state=torch.get_rng_state().clone()
    obj.update(model)
    assert torch.equal(rng_state,torch.get_rng_state())
    assert app.base.v21.state_hash(app.base.v21.model_checkpoint(model))==h
    assert obj.updates==1 and obj.eval_points==10201
    results['update_no_parameter_change_no_rng_consumption']=True
    # Hard constraints in the ORIGINAL source model (not a new implementation).
    t=torch.linspace(0,1,31);xx=torch.stack((t,torch.zeros_like(t)),1)
    with torch.no_grad():
        ic=(model(xx).flatten()-torch.sin(torch.pi*t)).abs().max().item()
        left=model(torch.stack((torch.zeros_like(t),t),1)).abs().max().item()
        right=model(torch.stack((torch.ones_like(t),t),1)).abs().max().item()
    assert max(ic,left,right)<2e-6
    results['hard_constraints_max_error']=max(ic,left,right)
    # Check restore-on-error of scoped proposal override.
    original=app.core.ResidualAdaptive2D
    try:
        with app.proposal_adapter():raise RuntimeError('intentional test')
    except RuntimeError:pass
    assert app.core.ResidualAdaptive2D is original
    results['scoped_adapter_restored_after_exception']=True
    return results


def integration_check():
    cpu=torch.device('cpu');cfg=app.base.pilot_config('smoke')
    old=app.core.ResidualAdaptive2D
    with tempfile.TemporaryDirectory(prefix='v234_control_') as tmp:
        tmp=Path(tmp);(tmp/'ref').mkdir();ref,_=app.core.get_reference(cfg,tmp/'ref')
        sg,_,_=app.base.v231.secondary_reference(cfg,ref,tmp/'ref')
        row1=app.base.run_candidate(app.base.CANDIDATE_VARIANT,'rad',cfg,ref,sg,cpu,tmp/'unwrapped','bc_feasible',seed=0)
        with app.proposal_adapter(factory=old) as observed:
            row2=app.base.run_candidate(app.base.CANDIDATE_VARIANT,'rad',cfg,ref,sg,cpu,tmp/'wrapped','bc_feasible',seed=0)
        numeric=['space_time_rel_l2','final_rel_l2','dense_physics_loss','selected_val_full_loss',
                 'independent_residual_rms','lbfgs_closure_calls']
        for k in numeric:assert row1[k]==row2[k],(k,row1[k],row2[k])
        for k in ['initial_state_sha256','warmup_state_sha256','sampling_sha256','selected_checkpoint']:
            assert row1[k]==row2[k],k
        a=torch.load(tmp/'unwrapped'/'final_model_state.pt',map_location='cpu',weights_only=True)
        b=torch.load(tmp/'wrapped'/'final_model_state.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(a[k],b[k]) for k in a)
    return {'adapter_old_proposal_final_weights_bitwise_equal_CPU':True,
            'optimizer_output_and_seed_path_control_equal':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--integration',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    result={'environment':app.core.runtime_info(torch.device('cpu')),'checks':run_checks()}
    if args.integration:result['integration']=integration_check()
    result['passed']=True
    print(json.dumps(result,indent=2))
    if args.output:app.write_json(args.output,result)

if __name__=='__main__':main()
