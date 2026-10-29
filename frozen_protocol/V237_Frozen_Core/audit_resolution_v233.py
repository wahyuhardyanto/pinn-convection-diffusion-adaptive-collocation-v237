#!/usr/bin/env python3
"""Read-only post-hoc residual/proposal-resolution audit for frozen V2.3.3 PINNs.
No training, optimizer, checkpoint selection or FDM/SG recomputation is performed.
Python >=3.9; numpy, pandas, torch, matplotlib. CPU float64 quadrature; float32 controls.
Only state_dict checkpoints are loaded, with torch.load(weights_only=True).
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, platform, re, sys, time, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

VERSION='1.0.1'
C,D,ETA=1.0,0.001,0.1
METHODS=('mc','lhs','rad','alhs')

class FrozenPINN(nn.Module):
    """Exact forward expressions from BoundaryLayerPINN and HardBothBoundaryLayerPINN."""
    def __init__(self, state, formulation, dtype=torch.float64):
        super().__init__()
        layers=[state['net.0.weight'].shape[1]]
        indices=sorted(int(k.split('.')[1]) for k in state if k.endswith('.weight'))
        blocks=[]
        for n,idx in enumerate(indices):
            w=state[f'net.{idx}.weight'];layers.append(w.shape[0])
            blocks.append(nn.Linear(w.shape[1],w.shape[0]))
            if n<len(indices)-1:blocks.append(nn.Tanh())
        if layers != [2,64,64,64,1]:raise ValueError(f'Unexpected architecture: {layers}')
        self.net=nn.Sequential(*blocks)
        self.formulation=formulation;self.delta=D/C
        self.to(dtype=dtype);self.load_state_dict(state,strict=True)
        self.requires_grad_(False);self.eval()
    def forward(self, xt):
        x,t=xt[:,:1],xt[:,1:]
        right=torch.cat((torch.ones_like(x),t),dim=1)
        floor=math.exp(-1/self.delta)
        b=(torch.exp((x-1)/self.delta)-floor)/(1-floor)
        correction=self.net(xt)-b*self.net(right)
        if self.formulation=='main':correction=x*correction
        return torch.sin(torch.pi*x)+t*correction

def residual(model, xy, chunk=2048):
    dtype=next(model.parameters()).dtype
    output=[]
    with torch.enable_grad():
        for a in range(0,len(xy),chunk):
            z=torch.as_tensor(xy[a:a+chunk],dtype=dtype).clone().requires_grad_(True)
            u=model(z)
            du=torch.autograd.grad(u,z,torch.ones_like(u),create_graph=True)[0]
            dxx=torch.autograd.grad(du[:,:1],z,torch.ones_like(du[:,:1]),create_graph=False)[0][:,:1]
            r=du[:,1:]+C*du[:,:1]-D*dxx
            output.append(r.detach().cpu().numpy().ravel().astype(np.float64))
    r=np.concatenate(output)
    if not np.isfinite(r).all():raise FloatingPointError('Nonfinite residual')
    return r

def digest(data):return hashlib.sha256(data).hexdigest()
def jsonwrite(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding='utf-8')

def checkpoints(archive,kind):
    """Read selected final checkpoints only; ignore smoke and Mac metadata."""
    records=[]
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith('/final_model_state.pt') or '/smoke/' in name or '__MACOSX' in name:continue
            if kind=='v233':
                if '/main_pilot/' in name:form='main'
                elif '/sensitivity_pilot/' in name:form='sensitivity'
                else:continue
            else:
                if '/validation/' not in name:continue
                form='main' if kind=='v232' else 'sensitivity'
            match=re.search(r'/([^/]+)_nf500_seed([0-4])/',name)
            if match is None:continue
            method,seed=match.group(1),int(match.group(2))
            if method not in METHODS:raise ValueError(name)
            data=z.read(name)
            state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
            records.append(dict(key=f'{form}_{method}_seed{seed}',formulation=form,method=method,seed=seed,
                archive=Path(archive).name,member=name,checkpoint_sha256=digest(data),state=state))
    return records

def midpoint_grid(n,offset_x=.5,offset_t=.5):
    x=(np.arange(n)+offset_x)/n;t=(np.arange(n)+offset_t)/n
    xx,tt=np.meshgrid(x,t,indexing='xy')
    return np.column_stack((xx.ravel(),tt.ravel()))

def graded_edges(n):
    """Symmetric geometry-only grading. Exactly n cells; no learned-grid tuning."""
    boundary={51:12,101:25,201:50}[n]
    core=n-2*boundary
    left=np.r_[0.0,np.geomspace(1e-6,.02,boundary)]
    return np.r_[left,np.linspace(.02,.98,core+1)[1:],(1-left[-2::-1])]

def proposal_grid(n,graded=False):
    ex=graded_edges(n) if graded else np.linspace(0,1,n+1)
    et=np.linspace(0,1,n+1)
    x=(ex[:-1]+ex[1:])/2;t=(et[:-1]+et[1:])/2
    xx,tt=np.meshgrid(x,t,indexing='xy')
    weights=np.outer(np.diff(et),np.diff(ex))
    assert len(ex)==n+1 and np.all(np.diff(ex)>0) and np.isclose(weights.sum(),1)
    return ex,et,np.column_stack((xx.ravel(),tt.ravel())),weights

def quadrature(order):
    """Composite Gauss-Legendre with exactly integrated physical cell weights.
    Breakpoints include all reported boundary bands and t=0.01,0.1.
    No evaluation at singular/incompatible corners.
    """
    left=np.unique(np.r_[0.,1e-4,.001,np.linspace(.001,.01,5),np.linspace(.01,.02,3)])
    ex=np.unique(np.r_[left,np.linspace(.02,.98,49),1-left[::-1]])
    et=np.unique(np.r_[0.,.001,.005,.01,np.linspace(.01,.1,7),np.linspace(.1,1,31)])
    g,w=np.polynomial.legendre.leggauss(order)
    def rule(e):
        mid=(e[:-1]+e[1:])/2;half=np.diff(e)/2
        nodes=(mid[:,None]+half[:,None]*g).ravel()
        weights=(half[:,None]*w).ravel()
        return nodes,weights
    x,wx=rule(ex);t,wt=rule(et)
    xx,tt=np.meshgrid(x,t,indexing='xy');ww=np.outer(wt,wx)
    assert np.isclose(ww.sum(),1)
    return x,t,np.column_stack((xx.ravel(),tt.ravel())),ww

BANDS=[('right_001',.999,1.),('right_0001',.9999,1.),('right_01',.99,1.),
       ('left_001',0.,.001),('left_0001',0.,.0001),('left_01',0.,.01)]

def summary_integrals(x,t,r,ww):
    r=r.reshape(len(t),len(x))
    I2=float(np.sum(ww*r*r));I1=float(np.sum(ww*np.abs(r)))
    out=dict(mean_r2=I2,rms=math.sqrt(I2),mean_abs_r=I1,n_points=r.size)
    for label,a,b in BANDS:
        mask=(x>=a)&(x<=b);mass=float(ww[:,mask].sum());local=float(np.sum(ww[:,mask]*r[:,mask]**2))
        out[f'{label}_rms']=math.sqrt(local/mass)
        out[f'{label}_r2_fraction']=local/I2
        q=ETA+(1-ETA)*np.abs(r)/(I1+1e-12)
        q=q/np.sum(ww*q)
        out[f'{label}_proposal_prob']=float(np.sum(ww[:,mask]*q[:,mask]))
        tm=t>=.01
        a2=float(np.sum(ww[tm][:,mask]*r[tm][:,mask]**2))
        out[f'{label}_rms_tge001']=math.sqrt(a2/ww[tm][:,mask].sum())
    for tag,a,b in [('early',0.,.01),('mid',.01,.1),('late',.1,1.)]:
        tm=(t>=a)&(t<=b)
        out[f'{tag}_r2_fraction']=float(np.sum(ww[tm]*r[tm]**2)/I2)
    return out

def proposal_stats(ex,et,r,ww,xq,tq,rq,wq):
    f=np.abs(r).reshape(ww.shape)
    mean=float(np.sum(ww*f))
    q=ETA+(1-ETA)*f/(mean+1e-12);q/=np.sum(q*ww)
    mean_ref=float(np.sum(wq*np.abs(rq)))
    qref=ETA+(1-ETA)*np.abs(rq)/(mean_ref+1e-12);qref/=np.sum(qref*wq)
    ix=np.clip(np.searchsorted(ex,xq,side='right')-1,0,len(ex)-2)
    it=np.clip(np.searchsorted(et,tq,side='right')-1,0,len(et)-2)
    field=q[np.ix_(it,ix)]
    # TV is a numerical integral on the fixed high-order quadrature; density jumps
    # need not align with its panels, so treat this as a diagnostic approximation.
    out=dict(proposal_mean_abs_r=mean,proposal_tv_approx=float(.5*np.sum(wq*np.abs(field-qref))))
    for label,a,b in BANDS:
        overlap=np.maximum(0,np.minimum(ex[1:],b)-np.maximum(ex[:-1],a))
        out[f'{label}_prob']=float(np.sum(q*np.outer(np.diff(et),overlap)))
    naive_f=ETA+(1-ETA)*f/(f.mean()+1e-12)
    naive_m=naive_f/naive_f.sum() # WRONG for nonuniform cells; measured as caution only.
    frac=np.maximum(0,np.minimum(ex[1:],1)-np.maximum(ex[:-1],.999))/np.diff(ex)
    out['wrong_equal_cell_mass_right_001_prob']=float(np.sum(naive_m*frac[None,:]))
    return out

def load_nominal(archive):
    frames=[]
    with zipfile.ZipFile(archive) as z:
        for form in ['main','sensitivity']:
            name=next(n for n in z.namelist() if n.endswith(f'/{form}_pilot/results_combined.csv'))
            df=pd.read_csv(io.BytesIO(z.read(name)));df['formulation']=form;frames.append(df)
    return pd.concat(frames,ignore_index=True)

def audit(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'audit_integrity.json').exists() and not args.overwrite:
        raise FileExistsError('Audit already complete. Use a fresh output folder or explicit --overwrite.')
    if (out/'checkpoint_resolution.csv').exists() and not (args.resume or args.overwrite):
        raise FileExistsError('Partial results exist. Use --resume or choose a fresh output folder.')
    torch.set_num_threads(args.threads)
    files=[(args.v233,'v233'),(args.v232,'v232'),(args.v231,'v231')]
    models=[];sources=[]
    for filename,kind in files:
        if filename:
            p=Path(filename)
            sources.append(dict(archive=p.name,sha256=digest(p.read_bytes())))
            models+=checkpoints(p,kind)
    if args.limit:models=models[:args.limit]
    keys=[m['key'] for m in models]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate checkpoint identities')
    expected={f'{f}_{m}_seed{s}' for f in ['main','sensitivity'] for m in METHODS for s in range(5)}
    nominal=load_nominal(args.v233)
    if not args.limit and set(keys)!=expected: print('WARNING: partial coverage',sorted(expected-set(keys)),flush=True)
    orders=[int(a) for a in args.orders.split(',')]
    if len(orders)<2:raise ValueError('Use >=2 quadrature orders for resolution comparison')
    grids={order:quadrature(order) for order in orders}
    plan=dict(version=VERSION,sources=sources,keys=sorted(keys),orders=orders,threads=args.threads,chunk=args.chunk,
              script_sha256=digest(Path(__file__).read_bytes()))
    plan_path=out/'audit_plan.json'
    if args.resume and (out/'checkpoint_resolution.csv').exists():
        if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise ValueError('Resume sources, code, or settings differ; use a fresh output folder.')
    jsonwrite(plan_path,plan)
    qrows=[];prows=[];rows=[];control=[];inventory=[]
    if args.resume and (out/'checkpoint_resolution.csv').exists():
        rows=pd.read_csv(out/'checkpoint_resolution.csv').to_dict('records')
        qrows=pd.read_csv(out/'quadrature_convergence.csv').to_dict('records')
        prows=pd.read_csv(out/'proposal_resolution.csv').to_dict('records')
        control=pd.read_csv(out/'precision_controls.csv').to_dict('records')
    done={r['key'] for r in rows}
    if not done.issubset(set(keys)):raise ValueError('Resume output contains unexpected checkpoint identities')
    inventory=[{k:v for k,v in item.items() if k!='state'} for item in models]
    tstart=time.perf_counter()
    for i,item in enumerate(sorted(models,key=lambda r:r['key'])):
        if item['key'] in done:continue
        meta={k:item[k] for k in ('key','formulation','method','seed')}
        model=FrozenPINN(item['state'],item['formulation'])
        original={k:v.detach().clone() for k,v in model.state_dict().items()}
        coarse={};coarse_rs={}
        for name,n,ox,ot in [('val31',31,.5,.5),('dense41',41,.31,.67),('candidate51',51,.5,.5)]:
            points=midpoint_grid(n,ox,ot);r=residual(model,points,args.chunk)
            coarse[name]=dict(mean_r2=float(np.mean(r*r)),mean_abs_r=float(np.mean(np.abs(r))))
            coarse_rs[name]=r
        prev=None;current=None
        for order in orders:
            x,t,pts,w=grids[order];r=residual(model,pts,args.chunk).reshape(w.shape)
            stats=summary_integrals(x,t,r,w)
            cr=dict(**meta,gauss_order=order,**stats)
            if prev:
                cr['mean_r2_rel_change_prev']=abs(stats['mean_r2']/prev['mean_r2']-1)
                cr['mean_abs_r_rel_change_prev']=abs(stats['mean_abs_r']/prev['mean_abs_r']-1)
                cr['right001_prob_rel_change_prev']=abs(stats['right_001_proposal_prob']/prev['right_001_proposal_prob']-1)
            qrows.append(cr);prev=stats;current=cr
        result=dict(**meta,**{k:v for k,v in current.items() if k not in meta})
        for name,stats in coarse.items():
            result[f'{name}_mean_r2']=stats['mean_r2']
            result[f'{name}_underestimate_factor']=current['mean_r2']/stats['mean_r2']
            result[f'{name}_captured_r2_fraction']=stats['mean_r2']/current['mean_r2']
        for n,graded in [(51,False),(101,False),(51,True),(101,True),(201,True)]:
            ex,et,pts_g,wg=proposal_grid(n,graded)
            rg=coarse_rs['candidate51'] if n==51 and not graded else residual(model,pts_g,args.chunk)
            ps=proposal_stats(ex,et,rg,wg,x,t,r,w)
            ps.update(mean_r2=float(np.sum(wg*rg.reshape(wg.shape)**2)),
                      mean_r2_ratio_reference=float(np.sum(wg*rg.reshape(wg.shape)**2)/current['mean_r2']))
            name=('graded' if graded else 'uniform')+str(n)
            prows.append(dict(**meta,grid=name,n_points=len(pts_g),**ps))
            result[f'{name}_right001_prob']=ps['right_001_prob']
            result[f'{name}_tv_approx']=ps['proposal_tv_approx']
        # Numerical precision controls: same weights, same physical coordinates,
        # no retraining and no mutation. CPU float32 vs CPU float64.
        model32=FrozenPINN(item['state'],item['formulation'],torch.float32)
        test_grids={'val31':midpoint_grid(31),'candidate51':midpoint_grid(51)}
        xp=1-np.geomspace(1e-6,.02,121);tp=np.linspace(.01,1,41)
        a,b=np.meshgrid(xp,tp);test_grids['right_log_probe']=np.column_stack((a.ravel(),b.ravel()))
        for tag,pts in test_grids.items():
            rd=residual(model,pts,args.chunk);rf=residual(model32,pts,args.chunk)
            control.append(dict(**meta,grid=tag,rms_float64=float(np.sqrt(np.mean(rd*rd))),
                rms_float32=float(np.sqrt(np.mean(rf*rf))),
                residual_difference_rms=float(np.sqrt(np.mean((rf-rd)**2))),
                residual_difference_max=float(np.max(np.abs(rf-rd)))))
        base=nominal[(nominal.formulation==item['formulation'])&(nominal.method==item['method'])&(nominal.seed==item['seed'])]
        assert len(base)==1
        for col in ['space_time_rel_l2','final_rel_l2','dense_physics_loss','final_physics_val_loss','selected_stage','selected_checkpoint']:
            if col in base:result['historical_'+col]=base.iloc[0][col]
        if 'dense_physics_loss' in base:
            result['old_dense64_relative_difference_saved']=coarse['dense41']['mean_r2']/float(base.iloc[0].dense_physics_loss)-1
        if not all(torch.equal(original[k],v) for k,v in model.state_dict().items()):raise AssertionError('Weight mutation')

        if item['key'] in ['main_mc_seed0','main_alhs_seed4','sensitivity_lhs_seed0','sensitivity_rad_seed0']:
            np.savez_compressed(out/(item['key']+'_audit_arrays.npz'),x=x,t=t,residual=r,area_weights=w,
                val31_xy=midpoint_grid(31),candidate51_xy=midpoint_grid(51),
                val31_residual=coarse_rs['val31'],candidate51_residual=coarse_rs['candidate51'])
        rows.append(result)
        # Write completed audit rows incrementally; no source files are modified.
        pd.DataFrame(rows).to_csv(out/'checkpoint_resolution.csv',index=False)
        pd.DataFrame(qrows).to_csv(out/'quadrature_convergence.csv',index=False)
        pd.DataFrame(prows).to_csv(out/'proposal_resolution.csv',index=False)
        pd.DataFrame(control).to_csv(out/'precision_controls.csv',index=False)
        print(f'[{i+1}/{len(models)}] {item["key"]}: J={current["mean_r2"]:.6g}; J/Jval31={result["val31_underestimate_factor"]:.2f}; conv={current.get("mean_r2_rel_change_prev",float("nan")):.2g}; elapsed={time.perf_counter()-tstart:.1f}s',flush=True)
    pd.DataFrame(inventory).to_csv(out/'checkpoint_inventory.csv',index=False)
    df=pd.DataFrame(rows)
    nums=df.select_dtypes(include=np.number).columns.difference(['seed','n_points','gauss_order'])
    df.groupby(['formulation','method'])[nums].median().reset_index().to_csv(out/'group_medians.csv',index=False)
    pp=pd.DataFrame(prows)
    pp.groupby(['formulation','method','grid']).median(numeric_only=True).reset_index().to_csv(out/'proposal_group_medians.csv',index=False)
    # Distinct rankings of FINAL MODELS only; no optimizer checkpoint reselection.
    ranked=df[['key','formulation','method','seed','val31_mean_r2','mean_r2']].copy()
    ranked['rank_val31']=ranked.groupby(['formulation','seed'])['val31_mean_r2'].rank(method='average')
    ranked['rank_resolved']=ranked.groupby(['formulation','seed'])['mean_r2'].rank(method='average')
    ranked.to_csv(out/'final_model_residual_ranks.csv',index=False)
    jsonwrite(out/'audit_integrity.json',dict(version=VERSION,training_runs=0,source_checkpoints_unchanged=True,
        evaluated_checkpoints=len(rows),expected_checkpoints=40,complete=set(keys)==expected,
        missing=sorted(expected-set(keys)),sources=sources,
        cpu_runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,python=sys.version,
                         platform=platform.platform(),threads=args.threads,dtype='float64',float32_controls=True),
        quadrature_orders=orders,quadrature_n_points={str(k):len(v[2]) for k,v in grids.items()},
        max_last_J_relative_change=float(df['mean_r2_rel_change_prev'].max()),
        max_last_absres_relative_change=float(df['mean_abs_r_rel_change_prev'].max()),
        max_last_right_probability_relative_change=float(df['right001_prob_rel_change_prev'].max()),
        elapsed_current_session_seconds=time.perf_counter()-tstart,
        scope='Selected final checkpoints only; no historical candidate-state weights or proposal history reconstructed.',
        proposal='Counterfactual diagnostic density at each final model; eta=0.1, residual absolute value. Physical cell areas included.',
        integral='J = integral_0^1 integral_0^1 r(x,t)^2 dx dt; no PDE/BC training weights applied.',
        script_sha256=digest(Path(__file__).read_bytes())))
    return df

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v233',required=True,help='PDE_v233_pilot_results.zip')
    p.add_argument('--v232',help='PDE_v232_ablation_results.zip; supplies 5 historical main MC checkpoints')
    p.add_argument('--v231',help='PDE_v231_validation_results(1).zip; supplies 5 historical sensitivity MC checkpoints')
    p.add_argument('--output',default='V233_Resolution_Audit_results')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--chunk',type=int,default=2048)
    p.add_argument('--orders',default='4,6,8')
    p.add_argument('--limit',type=int,default=0,help='Optional small pipeline test, not complete audit')
    p.add_argument('--resume',action='store_true',help='Continue a partial audit with unchanged sources and settings')
    p.add_argument('--overwrite',action='store_true',help='Overwrite audit outputs only, never source archives')
    args=p.parse_args();audit(args)
if __name__=='__main__':main()
