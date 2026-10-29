#!/usr/bin/env python3
"""V2.3.5 controlled training-objective diagnostic using identical archived tapes.

Default pilot: 2 samplers x 2 objectives x seed 0 = FOUR new runs.
Ablation: same experiment x seeds 0..4 = TWENTY total (pilot seed 0 reused).
This is a replay mechanism test, NOT a new online adaptive-sampler benchmark.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time
import sys
import numpy as np
import pandas as pd
import torch
import convection_diffusion_pilot_v233 as base
import convection_diffusion_proposal_v234 as v234
from objective_control_v235 import ObjectiveContext,sha_points
from objective_trainer_v235 import run_objective_candidate
from shared_transition_v235 import SharedTransition,FIX_VERSION

VERSION='2.3.5-sharedstart1'
HERE=Path(__file__).resolve().parent
METHODS=('rad','alhs')
OBJECTIVES=('plain','importance_uniform')
METRICS=('space_time_rel_l2','final_rel_l2','pinn_vs_sg_space_time_rel_l2',
         'resolved_J','resolved_right_001_rms','resolved_right_0001_rms',
         'selected_bc_max_abs','runtime_training_s')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,record):v234.write_json(Path(path),record)

def verify_sources():
    info=json.loads((HERE/'Replay_Provenance_V235.json').read_text())
    for name,digest in {**info['frozen_modules'],**info['files']}.items():
        if not (HERE/name).is_file() or sha(HERE/name)!=digest:
            raise RuntimeError('Frozen source missing or changed: '+name)
    return info


def configuration(preset):
    source='smoke' if preset=='smoke' else 'ablation'
    manifest=json.loads((HERE/'replay_tapes'/source/'source_manifest.json').read_text())
    # Config is the exact dict used for V234, without hyperparameter tuning here.
    return copy.deepcopy(manifest['pde_config']),manifest,source


def references(cfg):
    ref_keys=('c','D','x_range','t_range','reference_nx','reference_nt','reference_snapshots')
    signature=hashlib.sha256(json.dumps({k:cfg[k] for k in ref_keys},sort_keys=True).encode()).hexdigest()[:12]
    out=[]
    for prefix in ('fdm_reference_','sg_secondary_'):
        p=HERE/'frozen_references'/(prefix+signature+'.npz')
        with np.load(p,allow_pickle=False) as z:out.append({k:z[k].copy() for k in z.files})
    if not np.array_equal(out[0]['x'],out[1]['x']) or not np.array_equal(out[0]['t'],out[1]['t']):
        raise RuntimeError('Frozen references are not aligned')
    return tuple(out)


@contextmanager
def observe_states():
    original=base.v21.model_checkpoint
    states=[]
    def capture(model):
        state=original(model);states.append(state);return state
    base.v21.model_checkpoint=capture
    try:yield states
    finally:base.v21.model_checkpoint=original


def source_code_hashes():
    return {p.name:sha(p) for p in HERE.glob('*.py')}


def environment_diff(old,now):
    keys=('python','torch','numpy','scipy','pandas','device','dtype')
    return {k:{'source':old.get(k),'current':now.get(k)} for k in keys if old.get(k)!=now.get(k)}


def check_existing(run,which):
    target=run/which
    if not target.exists():return False
    rec=json.loads(target.read_text())
    for name,digest in rec['files'].items():
        p=run/name
        if not p.is_file() or sha(p)!=digest:
            raise RuntimeError('Recorded completed output changed: '+str(p))
    return True


def execute_run(folder,preset,method,objective,seed,cfg,device,env_matched):
    run=folder/'runs'/f'{method}_{objective}_seed{seed:03d}'
    if check_existing(run,'completion.json'):
        print('SKIP',run.name,flush=True);return
    source='smoke' if preset=='smoke' else 'ablation'
    tape=HERE/'replay_tapes'/source/f'{method}_seed{seed:03d}.npz'
    nf=cfg['n_collocation'][0]
    diag=run/'diagnostics'/f'{method}_nf{nf}_seed{seed}'
    if check_existing(run,'training_complete.json'):
        row=json.loads((run/'training_result.json').read_text())
        print('RESUME audits only',run.name,flush=True)
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError('Partial run found. Preserve/move only this run before retry: '+str(run))
        run.mkdir(parents=True)
        print('START',run.name,'same historical points; objective =',objective,flush=True)
        ctx=ObjectiveContext(tape,objective,cfg,gradient_audit=True)
        signature=dict(fix_version=FIX_VERSION,method=method,seed=int(seed),cfg=cfg,
            replay_tape_sha256=sha(tape),runtime=base.core.runtime_info(device),
            threads=torch.get_num_threads(),source_code_sha256=source_code_hashes())
        ctx.shared_transition=SharedTransition(
            folder/'shared_transition'/f'{method}_seed{seed:03d}',signature,device,objective)
        ref,sg=references(cfg)
        with observe_states() as states:
            row=run_objective_candidate(base.CANDIDATE_VARIANT,method,cfg,ref,sg,device,diag,
                                       'bc_feasible',seed=seed,objective_context=ctx)
        ctx.save(diag)
        row.update(ctx.shared_start_audit)
        cand=pd.read_csv(diag/'checkpoint_candidates.csv')
        if len(states)!=len(cand)+2:raise RuntimeError('Unexpected candidate observer count')
        saved=diag/'candidate_states';saved.mkdir(exist_ok=True)
        inventories=[]
        for state,(_,r) in zip(states[2:],cand.iterrows()):
            p=saved/(str(r.checkpoint_id)+'.pt');torch.save(state,p)
            inventories.append(dict(checkpoint_id=r.checkpoint_id,file_sha256=sha(p),
                                    state_sha256=base.v21.state_hash(state)))
        pd.DataFrame(inventories).to_csv(diag/'candidate_state_inventory.csv',index=False)
        hist=pd.read_csv(HERE/'replay_tapes'/source/'historical_results.csv')
        hist=hist[(hist.method==method)&(hist.seed==seed)].iloc[0]
        historical_match=row['sampling_sha256']==hist.sampling_sha256
        if not historical_match:raise RuntimeError('Replayed training points do not match V234')
        row.update(objective_version=VERSION,objective_mode=objective,proposal_mode='replay_v234',
             replay_family=source,proposal_profile='graded101',target_measure=('donor_q_per_set' if objective=='plain' else 'physical_uniform'),
             replay_tape_sha256=sha(tape),historical_sampling_hash_matches=historical_match,
             transition_state_sha256=ctx.transition_hash,
             replay_proposal_updates=ctx.proposal.updates,adaptive_updates=0,
             adaptive_candidate_evaluations=0,online_proposal_evaluations=0,
             adaptive_candidate_xmax=float(ctx.proposal.candidates[:,0].max()),adaptive_cell_width=np.nan,
             generation_pdf_frozen=True,importance_self_normalized=False,importance_clipped=False,
             lbfgs_weights_policy=('unweighted_unique' if objective=='plain' else 'per_generation_pdf_aggregate_duplicates'),
             lbfgs_duplicate_occurrences=ctx.lbfgs_duplicate_count,
             checkpoint_rule_changed=False,validation_uses_training_importance_weights=False,
             training_importance_weighted=objective=='importance_uniform',environment_matched_baseline=env_matched,
             candidate_states_saved=len(cand),gradient_audit_calls=len(ctx.gradlogs),
             replay_runtime_not_comparable_to_online_proposal=True)
        # MPS repetition need not be bitwise identical to its old session.
        controls={key:row[key]==hist[key] for key in ('initial_state_sha256','warmup_state_sha256','sampling_sha256')}
        write_json(run/'historical_controls.json',{'checks':controls,'environment_matched':env_matched,
            'finite_precision_cross_session_identity_not_guaranteed':True})
        write_json(run/'training_result.json',row)
        required=[run/'training_result.json',run/'historical_controls.json',diag/'final_model_state.pt',
                  diag/'optimizer_objective_trace.csv',diag/'same_state_gradient_comparison.csv',
                  diag/'replay_points_audit.csv',diag/'collocation_generation_weights.npz',diag/'joint_start_control.json']+list(saved.glob('*.pt'))
        write_json(run/'training_complete.json',{'files':{str(p.relative_to(run)):sha(p) for p in required}})
    # Audits occur ONLY after source-compatible checkpoint selection.
    model=base.v232.load_candidate(cfg,device,diag/'final_model_state.pt')
    out=base.v231.dense_outflow_residual_audit(model,cfg,device,diag,181,61)
    left=base.v232.dense_inflow_residual_audit(model,cfg,device,diag,181,61)
    resolved=v234.resolved_audit(diag/'final_model_state.pt',cfg,diag/'resolved_posthoc',preset)
    row.update(residual_rms_area_999=out['residual_rms_area_999'],
        residual_rms_area_9999=out['residual_rms_area_9999'],
        residual_rms_inflow_x_le_001=left['rms_area_x_le_001'],
        resolved_J=resolved['result']['mean_r2'],resolved_order=resolved['result']['order'],
        resolved_converged=resolved['quadrature_converged'],
        resolved_right_001_rms=resolved['result']['right_001_rms'],
        resolved_right_0001_rms=resolved['result']['right_0001_rms'])
    plot_objective_trace(diag)
    base.core.atomic_csv(run/'results_raw.csv',pd.DataFrame([row]))
    required=[run/'results_raw.csv',diag/'resolved_posthoc'/'resolved_audit.json']
    write_json(run/'completion.json',{'complete':True,'files':{str(p.relative_to(run)):sha(p) for p in required}})
    print('DONE',run.name,'global L2=',row['space_time_rel_l2'],'resolved J=',row['resolved_J'],flush=True)


def plot_objective_trace(diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(diag/'optimizer_objective_trace.csv')
    fig,ax=plt.subplots(figsize=(8,4.4))
    ax.plot(d.index,d.ordinary_pde,label='ordinary sample mean')
    ax.plot(d.index,d.is_uniform_estimate,label='importance-corrected estimate')
    ax.set_yscale('log');ax.set_xlabel('Diagnostic record (not epoch)');ax.set_ylabel('PDE loss')
    ax.set_title('Two objectives on identical points/current network');ax.legend();fig.tight_layout()
    fig.savefig(diag/'objective_estimates.png',dpi=145);plt.close(fig)
    g=pd.read_csv(diag/'same_state_gradient_comparison.csv')
    fig,ax=plt.subplots(figsize=(8,4.4))
    ax.plot(g.index,g.cosine_plain_vs_is,'o-');ax.set_ylim(-1.05,1.05)
    ax.set_xlabel('Fresh set / first L-BFGS call');ax.set_ylabel('PDE gradient cosine')
    ax.set_title('Gradient directions: ordinary versus importance');fig.tight_layout()
    fig.savefig(diag/'gradient_direction.png',dpi=145);plt.close(fig)


def report(folder,preset):
    rows=[]
    for p in sorted((folder/'runs').glob('*/results_raw.csv')):
        if check_existing(p.parent,'completion.json') and check_existing(p.parent,'training_complete.json'):
            rows.append(pd.read_csv(p))
    d=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    seeds=(0,) if preset=='smoke' else tuple(range(5))
    expected={(m,o,s) for m in METHODS for o in OBJECTIVES for s in seeds}
    present=set() if d.empty else set(zip(d.method,d.objective_mode,d.seed.astype(int)))
    if not d.empty and d.duplicated(['method','objective_mode','seed']).any():raise RuntimeError('Duplicate rows')
    stage_expected = {(m,o,0) for m in METHODS for o in OBJECTIVES} if preset=='pilot' else expected
    write_json(folder/'integrity.json',{'version':VERSION,'requested_stage':preset,
         'expected_stage_runs':len(stage_expected),'stage_complete':stage_expected.issubset(present),
         'expected_final_runs':len(expected),'completed':len(d),
         'missing_final':[list(k) for k in sorted(expected-present)],
         'complete':stage_expected.issubset(present),'ablation_complete':expected==present,
         'scope':'replay diagnostic, not online adaptive superiority; no inferential p-values'})
    if d.empty:return
    base.core.atomic_csv(folder/'results_combined.csv',d)
    joined=d[d.objective_mode=='plain'].merge(d[d.objective_mode=='importance_uniform'],
        on=['method','seed'],suffixes=('_plain','_importance'),validate='one_to_one')
    out=joined[['method','seed']].copy()
    for name in ('initial_state_sha256','warmup_state_sha256','transition_state_sha256','transition_optimizer_sha256','transition_rng_sha256','transition_points_sha256','shared_checkpoint_sha256','sampling_sha256','replay_tape_sha256'):
        out[name+'_matches']=joined[name+'_plain']==joined[name+'_importance']
        if not out[name+'_matches'].all():raise RuntimeError('Paired replay control differs: '+name)
    summary=[]
    for metric in METRICS:
        out[metric+'_plain']=joined[metric+'_plain'];out[metric+'_importance']=joined[metric+'_importance']
        out[metric+'_difference']=joined[metric+'_importance']-joined[metric+'_plain']
        for method,b in out.groupby('method'):
            delta=b[metric+'_difference']
            summary.append(dict(method=method,metric=metric,n_pairs=len(b),median_plain=b[metric+'_plain'].median(),
                median_importance=b[metric+'_importance'].median(),median_paired_delta=delta.median(),n_better=int((delta<0).sum())))
    base.core.atomic_csv(folder/'paired_objectives.csv',out)
    base.core.atomic_csv(folder/'paired_summary.csv',pd.DataFrame(summary))
    history=pd.read_csv(HERE/'replay_tapes'/('smoke' if preset=='smoke' else 'ablation')/'historical_results.csv')
    histcomp=d[d.objective_mode=='plain'].merge(history,on=['method','seed'],suffixes=('_replay','_historical'),validate='one_to_one')
    # Historical comparison is labeled; do not combine it into homogeneous paired results.
    base.core.atomic_csv(folder/'plain_replay_vs_historical.csv',histcomp)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('space_time_rel_l2','final_rel_l2','resolved_J'):
        fig,ax=plt.subplots(figsize=(8,4.4))
        for (m,o),b in d.groupby(['method','objective_mode']):
            b=b.sort_values('seed');ax.plot(b.seed,b[metric],'-o',label=m+' '+o)
        ax.set_xlabel('Seed');ax.set_ylabel(metric);ax.set_yscale('log');ax.legend(fontsize=8)
        ax.set_title('Controlled fixed-donor replay (descriptive)');fig.tight_layout()
        fig.savefig(folder/(metric+'.png'),dpi=145);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preset',choices=('smoke','pilot','ablation','report'),default='smoke')
    ap.add_argument('--methods',default='rad,alhs')
    ap.add_argument('--objectives',default='plain,importance_uniform')
    ap.add_argument('--seeds',default=None)
    ap.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    ap.add_argument('--output-root',type=Path,default=Path('pde_v235_outputs'))
    ap.add_argument('--allow-environment-mismatch',action='store_true')
    ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args()
    if args.preset=='report':
        for kind in ('smoke','replay'):
            if (args.output_root/kind).exists():report(args.output_root/kind,'smoke' if kind=='smoke' else 'ablation')
        return
    info=verify_sources();cfg,old,source=configuration(args.preset)
    methods=[a.strip() for a in args.methods.split(',')]
    objectives=[a.strip() for a in args.objectives.split(',')]
    if not set(methods).issubset(METHODS) or not set(objectives).issubset(OBJECTIVES):raise ValueError('Unknown methods/objectives')
    s=args.seeds if args.seeds is not None else ('0-4' if args.preset=='ablation' else '0')
    seeds=list(range(int(s.split('-')[0]),int(s.split('-')[1])+1)) if '-' in s else [int(x) for x in s.split(',')]
    if not set(seeds).issubset(set([0]) if source=='smoke' else set(range(5))):raise ValueError('No frozen tapes for those seeds')
    if args.device=='auto':
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:device=torch.device(args.device)
    torch.set_num_threads(args.threads)
    runtime=base.core.runtime_info(device)
    diff=environment_diff(old['runtime'],runtime)
    if diff and source!='smoke' and not args.allow_environment_mismatch:
        raise RuntimeError('Environment differs from training baseline; use the frozen Mac kernel. Details: '+json.dumps(diff))
    if diff:print('Environment difference (explicitly recorded):',diff,flush=True)
    folder=args.output_root.resolve()/('smoke' if source=='smoke' else 'replay');folder.mkdir(parents=True,exist_ok=True)
    body=dict(version=VERSION,source_family=source,cfg=cfg,runtime=runtime,threads=args.threads,
              environment_difference=diff,source_code_sha256=source_code_hashes(),
              replay_provenance_sha256=sha(HERE/'Replay_Provenance_V235.json'),
              protocol='fixed donor replay; ordinary versus ordinary-IS; same old grid31 selection')
    p=folder/'manifest.json'
    if p.exists() and json.loads(p.read_text())!=body:raise RuntimeError('Manifest differs; use a NEW output root')
    if not p.exists():write_json(p,body)
    for method in methods:
        for seed in seeds:
            for objective in objectives:
                execute_run(folder,args.preset,method,objective,seed,cfg,device,not diff)
                report(folder,args.preset)
    print('RESULTS:',folder,flush=True)

if __name__=='__main__': main()
