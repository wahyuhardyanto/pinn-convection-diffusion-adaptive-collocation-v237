#!/usr/bin/env python3
"""V2.3.6: read-only frozen replay baseline -> ten controlled online runs.

Production never regenerates the original common transition. The original V235
trainer/loss/selection is reused, with OnlineObjectiveContext providing the only
algorithmic change: current-model proposal updates and fresh generated points.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import os
import shutil
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch
import convection_diffusion_objective_v235 as replay
import convection_diffusion_proposal_v234 as v234
import shared_transition_v235 as shared
from online_context_v236 import OnlineObjectiveContext, ReadOnlyTransition

VERSION='2.3.6-online-transfer1'
HERE=Path(__file__).resolve().parent
BASE=HERE/'frozen_replay_v235'
METHODS=('rad','alhs')
RUNTIME_KEYS=('python','torch','numpy','scipy','pandas','system','machine','device','dtype')
CONTROL_KEYS=('initial_state_sha256','warmup_state_sha256','shared_checkpoint_sha256',
              'shared_model_sha256','transition_state_sha256','transition_optimizer_sha256',
              'transition_rng_sha256','transition_points_sha256')
METRICS=('space_time_rel_l2','final_rel_l2','pinn_vs_sg_space_time_rel_l2',
         'resolved_J','resolved_right_001_rms','resolved_right_0001_rms',
         'selected_bc_max_abs','independent_residual_rms')
core=replay.base.core

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x): v234.write_json(Path(p),x)

def verify_package():
    p=json.loads((HERE/'Provenance_V236.json').read_text())
    for name,digest in {**p['source_code_preserved'],**p['frozen_files']}.items():
        if not (HERE/name).is_file() or sha(HERE/name)!=digest:
            raise RuntimeError('Frozen dependency/baseline modified: '+name)
    replay.verify_sources()
    return p

def configuration(stage):
    cfg,_,_=replay.configuration('smoke' if stage=='smoke' else 'ablation')
    if stage!='smoke':
        frozen=json.loads((BASE/'manifest.json').read_text())
        if cfg!=frozen['cfg']: raise RuntimeError('Training configuration drift')
    return cfg

def resolve_replay_root(path):
    path=Path(path).expanduser().resolve()
    if (path/'replay'/'manifest.json').is_file() or (path/'smoke'/'manifest.json').is_file(): return path
    if path.name=='replay' and (path/'manifest.json').is_file(): return path.parent
    raise FileNotFoundError('REPLAY_ROOT must be the completed V235 output root, containing '
        'replay/manifest.json and replay/shared_transition/. Not a code folder or summary TXT: '+str(path))

def preflight_source(root,device,stage,targets):
    """Inspect originals without writing; full transfer tied to frozen uploaded rows."""
    verify_package()
    root=resolve_replay_root(root)
    family='smoke' if stage=='smoke' else 'replay'
    folder=root/family
    mf=folder/'manifest.json';csv=folder/'results_combined.csv'
    if not mf.is_file() or not csv.is_file():
        raise FileNotFoundError('Original completed V235 '+family+' manifest/results are required: '+str(folder))
    manifest=json.loads(mf.read_text());cfg=configuration(stage)
    if manifest.get('cfg')!=cfg: raise RuntimeError('V235 config does not match the frozen '+family+' protocol')
    if manifest.get('version')!='2.3.5-sharedstart1': raise RuntimeError('V235 SharedStart results are required')
    current=core.runtime_info(device)
    diff={k:{'replay':manifest['runtime'].get(k),'current':current.get(k)} for k in RUNTIME_KEYS
          if manifest['runtime'].get(k)!=current.get(k)}
    if manifest.get('threads')!=torch.get_num_threads():
        diff['threads']={'replay':manifest.get('threads'),'current':torch.get_num_threads()}
    if diff: raise RuntimeError('Environment differs; use the original kernel, not an override: '+json.dumps(diff))
    expected_source=json.loads((BASE/'manifest.json').read_text())['source_code_sha256']
    for name,digest in expected_source.items():
        if manifest['source_code_sha256'].get(name)!=digest:
            raise RuntimeError('V235 source identity differs: '+name)
    local=pd.read_csv(csv)
    if local.duplicated(['method','objective_mode','seed']).any(): raise RuntimeError('Duplicate source replay rows')
    importance=local[local.objective_mode=='importance_uniform'].copy()
    frozen=pd.read_csv(BASE/'importance_baseline10.csv') if stage!='smoke' else importance.copy()
    if stage!='smoke':
        expected={(m,s) for m in METHODS for s in range(5)}
        if set(zip(importance.method,importance.seed.astype(int)))!=expected:
            raise RuntimeError('The complete ten historical importance rows are required before online transfer')
        if set(zip(local.method,local.objective_mode,local.seed.astype(int)))!={(m,o,s) for m in METHODS for o in ('plain','importance_uniform') for s in range(5)}:
            raise RuntimeError('Expected all 20 V235 ablation results, without extra/missing rows')
    if stage!='smoke':
        frozen_manifest=json.loads((BASE/'manifest.json').read_text())
        if manifest['runtime']!=frozen_manifest['runtime'] or manifest['threads']!=frozen_manifest['threads']:
            raise RuntimeError('Local replay runtime metadata differs from the frozen uploaded ablation')
        frozen_all=pd.read_csv(BASE/'results_combined.csv').set_index(['method','objective_mode','seed'])
        for _,r in local.iterrows():
            original=frozen_all.loc[(r.method,r.objective_mode,int(r.seed))]
            for key in CONTROL_KEYS+('sampling_sha256','replay_tape_sha256'):
                if str(r[key])!=str(original[key]):
                    raise RuntimeError('Frozen 20-run identity mismatch: '+key)
            for key in METRICS:
                if not np.isclose(float(r[key]),float(original[key]),rtol=1e-12,atol=1e-15):
                    raise RuntimeError('Frozen 20-run metric changed: '+key)
    records=[];lookup={}
    for method,seed in targets:
        a=importance[(importance.method==method)&(importance.seed==seed)]
        b=frozen[(frozen.method==method)&(frozen.seed==seed)]
        if len(a)!=1 or len(b)!=1: raise RuntimeError('Missing historical importance run '+method+' seed '+str(seed))
        row=a.iloc[0];expected=b.iloc[0]
        for key in CONTROL_KEYS+('sampling_sha256','replay_tape_sha256'):
            if str(row[key])!=str(expected[key]): raise RuntimeError('Frozen replay identity mismatch: '+key)
        for key in METRICS:
            if not np.isclose(float(row[key]),float(expected[key]),rtol=1e-12,atol=1e-15):
                raise RuntimeError('Frozen replay metric changed: '+key)
        if not bool(row.get('environment_matched_baseline',False)) and stage!='smoke':
            raise RuntimeError('Historical production run recorded an environment mismatch')
        run=folder/'runs'/f'{method}_importance_uniform_seed{seed:03d}'
        if not replay.check_existing(run,'completion.json') or not replay.check_existing(run,'training_complete.json'):
            raise RuntimeError('Historical replay run is not complete: '+str(run))
        source_dir=folder/'shared_transition'/f'{method}_seed{seed:03d}'
        state=source_dir/'common_transition.pt';meta_path=source_dir/'manifest.json'
        if not state.is_file() or not meta_path.is_file():
            raise FileNotFoundError('Do not regenerate the prefix. Missing original shared transition: '+str(source_dir))
        meta=json.loads(meta_path.read_text());sig=meta['signature']
        if sha(state)!=row.shared_checkpoint_sha256 or sha(state)!=meta['file_sha256']:
            raise RuntimeError('Source transition file checksum mismatch: '+str(state))
        if sig['cfg']!=cfg or sig['method']!=method or int(sig['seed'])!=seed:
            raise RuntimeError('Shared transition signature differs from source run')
        if sig['runtime']!=manifest['runtime'] or sig['threads']!=manifest['threads']:
            raise RuntimeError('Shared transition runtime differs from source manifest')
        tape=HERE/'replay_tapes'/('smoke' if stage=='smoke' else 'ablation')/f'{method}_seed{seed:03d}.npz'
        if sha(tape)!=row.replay_tape_sha256 or sig['replay_tape_sha256']!=sha(tape):
            raise RuntimeError('Transition design provenance mismatch')
        bundle=torch.load(state,map_location='cpu',weights_only=True)
        for key in ('model','optimizer','rng','warm_state'):
            if shared.tree_hash(bundle[key])!=meta['content_hashes'][key]:
                raise RuntimeError('Shared transition content corrupted: '+key)
        actual={'initial_state_sha256':bundle['initial_state_sha256'],
            'warmup_state_sha256':replay.base.v21.state_hash(bundle['warm_state']),
            'shared_checkpoint_sha256':sha(state),'shared_model_sha256':shared.tree_hash(bundle['model']),
            'transition_state_sha256':replay.base.v21.state_hash(bundle['model']),
            'transition_optimizer_sha256':shared.tree_hash(bundle['optimizer']),
            'transition_rng_sha256':shared.tree_hash(bundle['rng']),
            'transition_points_sha256':bundle['transition_points_sha256']}
        for key in CONTROL_KEYS:
            if str(actual[key])!=str(row[key]): raise RuntimeError('Shared state vs result mismatch: '+key)
        with np.load(tape,allow_pickle=False) as z:
            if replay.sha_points(z['points'][0])!=bundle['transition_points_sha256']:
                raise RuntimeError('Transition points fail source hash check')
        rec=dict(method=method,seed=int(seed),source_folder=str(source_dir),
                 state_sha256=sha(state),manifest_sha256=sha(meta_path),
                 all_control_checks_passed=True)
        records.append(rec);lookup[(method,seed)]={'row':row.to_dict(),'folder':source_dir,'tape':tape}
    return dict(root=root,folder=folder,cfg=cfg,manifest=manifest,runtime=current,
                baseline=frozen,lookup=lookup,records=records)

@contextmanager
def output_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=root/'.v236.lock'
    try: fd=os.open(str(lock),os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError:
        raise RuntimeError('Output is locked. Do not run concurrent notebooks. Inspect '+str(lock)+'; remove only after confirming no process is active.')
    os.write(fd,('pid='+str(os.getpid())+'\n').encode());os.close(fd)
    try: yield
    finally: lock.unlink(missing_ok=True)

def make_manifest(folder,stage,info):
    family='smoke' if stage=='smoke' else 'online'
    body=dict(version=VERSION,family=family,cfg_legacy=info['cfg'],
        effective_protocol=dict(methods=list(METHODS),formulation='hard_both_bl_uniform',
            constraint_mode='hard_ic_hard_left_hard_right',objective='importance_uniform',
            manual_boundary_enrichment=False,proposal='graded101_current_model_online',
            checkpoint_selection='unchanged_grid31_bc_feasible',n_collocation=info['cfg']['n_collocation'][0],
            target_measure='physical_uniform',generation_pdf='pre-rounding mass / physical area',
            no_importance_clipping=True,no_self_normalization=True,
            initial_state_policy='load exact local V235 common transition; never regenerate',
            lBFGS_policy='frozen data, per-generation weights, sum duplicate contributions'),
        runtime=info['runtime'],threads=torch.get_num_threads(),
        source_replay_root=str(info['root']),source_manifest_sha256=sha(info['folder']/'manifest.json'),
        frozen_baseline_sha256=sha(BASE/'importance_baseline10.csv') if stage!='smoke' else sha(info['folder']/'results_combined.csv'),
        source_code_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))})
    path=folder/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=body:
        raise RuntimeError('Online manifest differs. Keep old outputs and use a new output root')
    if not path.exists(): write_json(path,body)
    dest=folder/'baseline_replay_frozen.csv'
    data=info['baseline'].to_csv(index=False)
    if dest.exists() and dest.read_text()!=data: raise RuntimeError('Frozen replay snapshot changed')
    if not dest.exists(): dest.write_text(data)
    return body

def save_candidates(diag,states,row):
    table=pd.read_csv(diag/'checkpoint_candidates.csv')
    if len(states)!=2+len(table): raise RuntimeError('Candidate observer inventory mismatch')
    saved=diag/'candidate_states';saved.mkdir(exist_ok=True)
    final=torch.load(diag/'final_model_state.pt',map_location='cpu',weights_only=True)
    inventory=[]
    for state,(_,r) in zip(states[2:],table.iterrows()):
        cid=str(r.checkpoint_id)
        if cid!='lbfgs' and not (cid.startswith('adam_') and cid[5:].isdigit()):
            raise RuntimeError('Invalid candidate ID')
        p=saved/(cid+'.pt');torch.save(state,p)
        inventory.append(dict(checkpoint_id=cid,file_sha256=sha(p),state_sha256=replay.base.v21.state_hash(state),selected=cid==row['selected_checkpoint']))
        if cid==row['selected_checkpoint'] and shared.tree_hash(state)!=shared.tree_hash(final):
            raise RuntimeError('Selected candidate and final model differ')
    pd.DataFrame(inventory).to_csv(diag/'candidate_state_inventory.csv',index=False)
    return len(table)

def run_one(folder,stage,method,seed,info,device):
    cfg=info['cfg'];source=info['lookup'][(method,seed)];old=source['row']
    run=folder/'runs'/f'{method}_seed{seed:03d}'
    if replay.check_existing(run,'completion.json'):
        if not replay.check_existing(run,'training_complete.json'): raise RuntimeError('Missing training marker')
        print('SKIP completed online',method,seed,flush=True);return
    diag=run/'diagnostics'/f'{method}_nf{cfg["n_collocation"][0]}_seed{seed}'
    if replay.check_existing(run,'training_complete.json'):
        row=json.loads((run/'training_result.json').read_text())
        print('RESUME post-hoc only',method,seed,flush=True)
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError('Partial training found. Preserve/move only this run; no mid-epoch resume: '+str(run))
        run.mkdir(parents=True,exist_ok=True)
        ctx=OnlineObjectiveContext(source['tape'],cfg,gradient_audit=True)
        ctx.shared_transition=ReadOnlyTransition(source['folder'],device,old['shared_checkpoint_sha256'])
        ref,sg=replay.references(cfg)
        print('START online',method,'seed',seed,'importance_uniform; prefix is read-only',flush=True)
        with replay.observe_states() as states:
            row=replay.run_objective_candidate(replay.base.CANDIDATE_VARIANT,method,cfg,ref,sg,device,diag,
                                               'bc_feasible',seed=seed,objective_context=ctx)
        ctx.save(diag)
        row.update(ctx.shared_start_audit)
        row['transition_state_sha256']=ctx.transition_hash
        controls={k:str(row[k])==str(old[k]) for k in CONTROL_KEYS}
        if not all(controls.values()): raise RuntimeError('Online/replay joint start differs: '+str(controls))
        n_candidates=save_candidates(diag,states,row)
        updates=(cfg['joint_epochs']-1)//cfg['resample_every']+1
        if ctx.proposal.updates!=updates or ctx.proposal.eval_points!=updates*cfg['adapt_grid_side']**2:
            raise RuntimeError('Expected actual online residual evaluations were not performed')
        row.update(protocol_version=VERSION,transfer_version=VERSION,method=method,
            objective_mode='importance_uniform',proposal_mode='online_current_model',
            proposal_profile='graded101',target_measure='physical_uniform',
            training_importance_weighted=True,importance_self_normalized=False,importance_clipped=False,
            generation_pdf_frozen_per_set=True,generation_pdf_from_current_model=True,
            lbfgs_weights_policy='per_generation_pdf_aggregate_duplicates',
            lbfgs_duplicate_occurrences=ctx.lbfgs_duplicate_count,
            online_proposal_evaluations=ctx.proposal.eval_points,adaptive_updates=ctx.proposal.updates,
            replay_proposal_updates=0,adaptive_cell_width=np.nan,
            adaptive_cell_width_min=float(ctx.proposal.dx.min()),adaptive_cell_width_max=float(ctx.proposal.dx.max()),
            adaptive_candidate_xmax=float(ctx.proposal.candidates[:,0].max()),
            proposal_residual_seconds=ctx.proposal.eval_seconds,
            checkpoint_rule_changed=False,validation_uses_training_importance_weights=False,
            environment_matched_baseline=True,candidate_states_saved=n_candidates,
            gradient_audit_calls=len(ctx.gradlogs),paired_start_verified=True,
            source_replay_sampling_sha256=old['sampling_sha256'],
            online_sampling_equals_replay=row['sampling_sha256']==old['sampling_sha256'],
            source_prefix_retrained=False,runtime_excludes_prefix_training=True)
        pairing=dict(method=method,seed=seed,controls=controls,all_matched=all(controls.values()),
                     point_sequences_after_transition_expected_to_differ=True,
                     source_transition_untouched=True,source_checkpoint=old['shared_checkpoint_sha256'])
        write_json(run/'pairing_audit.json',pairing)
        write_json(run/'training_result.json',row)
        required=[run/'training_result.json',run/'pairing_audit.json',diag/'final_model_state.pt',
                  diag/'online_points_audit.csv',diag/'collocation_generation_weights.npz',
                  diag/'online_proposal_history.npz',diag/'joint_start_control.json',
                  diag/'candidate_state_inventory.csv',diag/'optimizer_objective_trace.csv',
                  diag/'same_state_gradient_comparison.csv',diag/'online_proposal_trace.csv']+list((diag/'candidate_states').glob('*.pt'))
        write_json(run/'training_complete.json',{'files':{str(p.relative_to(run)):sha(p) for p in required}})
    model=replay.base.v232.load_candidate(cfg,device,diag/'final_model_state.pt')
    out=replay.base.v231.dense_outflow_residual_audit(model,cfg,device,diag,181,61)
    left=replay.base.v232.dense_inflow_residual_audit(model,cfg,device,diag,181,61)
    res=v234.resolved_audit(diag/'final_model_state.pt',cfg,diag/'resolved_posthoc',stage)
    row.update(residual_rms_area_999=out['residual_rms_area_999'],residual_rms_area_9999=out['residual_rms_area_9999'],
        residual_rms_inflow_x_le_001=left['rms_area_x_le_001'],
        resolved_J=res['result']['mean_r2'],resolved_order=res['result']['order'],
        resolved_converged=res['quadrature_converged'],
        resolved_right_001_rms=res['result']['right_001_rms'],resolved_right_0001_rms=res['result']['right_0001_rms'])
    replay.plot_objective_trace(diag)
    plot_proposal(diag)
    core.atomic_csv(run/'results_raw.csv',pd.DataFrame([row]))
    write_json(run/'completion.json',{'complete':True,'files':{str(p.relative_to(run)):sha(p) for p in [run/'results_raw.csv',diag/'resolved_posthoc'/'resolved_audit.json']}})
    print('DONE online',method,seed,'L2=',row['space_time_rel_l2'],'J=',row['resolved_J'],flush=True)

def plot_proposal(diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(diag/'online_proposal_trace.csv')
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(d.joint_epoch,d.n_x_ge_0999_actual,'o-',label='Actual samples')
    ax.plot(d.joint_epoch,d.expected_n_x_ge_0999,'s--',label='Expected under online q')
    ax.set(xlabel='Joint epoch at refresh',ylabel='Points at x >= 0.999',title='Current-model proposal coverage')
    ax.legend();fig.tight_layout();fig.savefig(diag/'online_proposal_coverage.png',dpi=150);plt.close(fig)

def report(folder,stage):
    folder=Path(folder);rows=[]
    for p in sorted((folder/'runs').glob('*/results_raw.csv')):
        if replay.check_existing(p.parent,'completion.json') and replay.check_existing(p.parent,'training_complete.json'):
            rows.append(pd.read_csv(p))
    d=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    seeds=(0,) if stage=='smoke' else tuple(range(5))
    expected={(m,s) for m in METHODS for s in seeds}
    present=set() if d.empty else set(zip(d.method,d.seed.astype(int)))
    if not present.issubset(expected) or (not d.empty and d.duplicated(['method','seed']).any()):
        raise RuntimeError('Duplicate/unexpected online results')
    if not d.empty:
        if not (d.proposal_mode=='online_current_model').all(): raise RuntimeError('Replay and online rows were mixed')
        if not d.paired_start_verified.all(): raise RuntimeError('Pairing controls failed')
    target={(m,0) for m in METHODS} if stage in ('smoke','pilot') else expected
    integrity=dict(version=VERSION,requested_stage=stage,completed_online=len(d),expected_online_total=len(expected),
        expected_stage_runs=len(target),stage_complete=target.issubset(present),
        online_complete=expected==present,missing_online=[list(x) for x in sorted(expected-present)],
        historical_replay_rows_not_counted_as_new=True,
        scope='paired replay-to-online transfer; not definitive sampler superiority; no inferential p-values')
    write_json(folder/'integrity.json',integrity)
    if d.empty:return integrity
    core.atomic_csv(folder/'results_online.csv',d)
    b=pd.read_csv(folder/'baseline_replay_frozen.csv')
    b=b[b.objective_mode=='importance_uniform']
    merged=b.merge(d,on=['method','seed'],suffixes=('_replay','_online'),validate='one_to_one')
    paired=merged[['method','seed']].copy()
    for k in CONTROL_KEYS:
        paired[k+'_matches']=merged[k+'_replay']==merged[k+'_online']
        if not paired[k+'_matches'].all():raise RuntimeError('Report pairing differs: '+k)
    stats=[]
    for metric in METRICS:
        old=merged[metric+'_replay'].astype(float);new=merged[metric+'_online'].astype(float)
        paired[metric+'_replay']=old;paired[metric+'_online']=new
        paired[metric+'_delta']=new-old
        paired[metric+'_ratio']=np.divide(new,old,out=np.full(len(old),np.nan),where=old!=0)
        for method,g in paired.groupby('method'):
            a=g[metric+'_replay'];z=g[metric+'_online'];delta=g[metric+'_delta']
            stats.append(dict(method=method,metric=metric,n_pairs=len(g),
                median_replay=a.median(),median_online=z.median(),median_paired_delta=delta.median(),
                median_paired_ratio=g[metric+'_ratio'].median(),n_better=int((delta<0).sum()),
                n_equal=int((delta==0).sum())))
    core.atomic_csv(folder/'paired_replay_online.csv',paired)
    core.atomic_csv(folder/'paired_summary.csv',pd.DataFrame(stats))
    # Preserve stage identification instead of treating data as one homogeneous sample.
    bx=b.copy();bx['experiment_stage']='frozen_replay_v235'
    dx=d.copy();dx['experiment_stage']='online_v236'
    core.atomic_csv(folder/'baseline_replay_plus_online.csv',pd.concat([bx,dx],ignore_index=True,sort=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=folder/'figures';figdir.mkdir(exist_ok=True)
    for metric in ('space_time_rel_l2','final_rel_l2','resolved_J'):
        fig,ax=plt.subplots(figsize=(8.4,4.7))
        for method in METHODS:
            g=paired[paired.method==method].sort_values('seed')
            if len(g):
                ax.plot(g.seed,g[metric+'_replay'],'o--',label=method.upper()+' replay')
                ax.plot(g.seed,g[metric+'_online'],'s-',label=method.upper()+' online')
        ax.set(xlabel='Seed',ylabel=metric,title='Importance objective: frozen replay vs online')
        ax.set_yscale('log');ax.legend();fig.tight_layout();fig.savefig(figdir/(metric+'.png'),dpi=155);plt.close(fig)
    return integrity

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset',choices=('preflight','smoke','pilot','online','report'),default='preflight')
    ap.add_argument('--preflight-stage',choices=('smoke','online'),default='online')
    ap.add_argument('--replay-root',type=Path)
    ap.add_argument('--output-root',type=Path,default=Path('pde_v236_outputs'))
    ap.add_argument('--methods',default='rad,alhs');ap.add_argument('--seeds',default=None)
    ap.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto');ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args()
    if args.preset=='report':
        for family in ('smoke','online'):
            folder=args.output_root/family
            if (folder/'manifest.json').exists():report(folder,'smoke' if family=='smoke' else 'online')
        return
    if args.replay_root is None:ap.error('--replay-root is required; use the completed V235 SharedStart OUTPUT_ROOT')
    methods=[x.strip() for x in args.methods.split(',')]
    if not methods or not set(methods).issubset(METHODS) or len(set(methods))!=len(methods):ap.error('Use rad, alhs, or rad,alhs')
    stage=args.preflight_stage if args.preset=='preflight' else args.preset
    s=args.seeds or ('0' if stage in ('smoke','pilot') else '0-4')
    seeds=list(range(int(s.split('-')[0]),int(s.split('-')[1])+1)) if '-' in s else [int(x) for x in s.split(',')]
    allowed={0} if stage in ('smoke','pilot') else set(range(5))
    if not seeds or not set(seeds).issubset(allowed) or len(set(seeds))!=len(seeds):ap.error('Invalid seeds for stage')
    targets=[(m,s) for m in methods for s in seeds]
    if args.device=='auto':device=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:device=torch.device(args.device)
    torch.set_num_threads(args.threads)
    with output_lock(args.output_root):
        info=preflight_source(args.replay_root,device,stage,targets)
        outroot=args.output_root.expanduser().resolve()
        if outroot==info['root'] or info['root'] in outroot.parents:
            raise RuntimeError('Use a separate V236 output root, outside the V235 source outputs')
        if outroot in info['root'].parents:raise RuntimeError('Output root must not enclose V235 source outputs')
        write_json(outroot/('preflight_'+stage+'.json'),dict(version=VERSION,passed=True,
            stage=stage,source_root=str(info['root']),runtime=info['runtime'],checked=len(targets),runs=info['records']))
        print('PREFLIGHT PASSED:',len(targets),'original starts; baseline retained read-only',flush=True)
        if args.preset=='preflight':return
        folder=outroot/('smoke' if stage=='smoke' else 'online');folder.mkdir(exist_ok=True)
        make_manifest(folder,stage,info)
        for method,seed in targets:
            run_one(folder,stage,method,seed,info,device)
            report(folder,stage)
        print('RESULTS:',folder,flush=True)

if __name__=='__main__':main()
