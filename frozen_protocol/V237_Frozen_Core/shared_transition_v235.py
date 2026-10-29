#!/usr/bin/env python3
"""Exact common joint-start checkpoint for V235 paired replay objectives.

No tolerance bypass. Model parameters, Adam state and RNG states are validated
bitwise after restoration. This is a new controlled-start protocol; old outputs
must remain separate. No pickle objects beyond weights_only-safe builtins/tensors.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch

FIX_VERSION = "V235-sharedstart-1"


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def tree_hash(value):
    """Content hash independent of tensor device and torch.save container bytes."""
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous()
            h.update(b'tensor');h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=lambda k:(type(k).__name__,repr(k))):
                visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:visit(x)
        else:
            h.update(type(v).__name__.encode());h.update(repr(v).encode())
        h.update(b'|')
    visit(value)
    return h.hexdigest()


def capture_rng(device, generators):
    out={'torch_cpu':torch.get_rng_state().clone(),
         'numpy_generators':{k:cpu_tree(g.bit_generator.state) for k,g in generators.items()},
         'python_random':random.getstate()}
    if device.type=='mps':
        out['torch_device']=torch.mps.get_rng_state().cpu().clone()
    elif device.type=='cuda':
        out['torch_device']=torch.cuda.get_rng_state(device).cpu().clone()
    return out


def restore_rng(saved, device, generators):
    torch.set_rng_state(saved['torch_cpu'])
    if set(generators)!=set(saved['numpy_generators']):
        raise RuntimeError('RNG generator inventory differs from the shared start')
    for k,g in generators.items():g.bit_generator.state=copy.deepcopy(saved['numpy_generators'][k])
    random.setstate(saved['python_random'])
    if device.type=='mps':torch.mps.set_rng_state(saved['torch_device'])
    elif device.type=='cuda':torch.cuda.set_rng_state(saved['torch_device'],device)


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SharedTransition:
    def __init__(self, folder, signature, device, creator_objective):
        self.folder=Path(folder);self.signature=signature;self.device=device
        self.creator_objective=creator_objective
        self.state_path=self.folder/'common_transition.pt'
        self.meta_path=self.folder/'manifest.json'
        self.bundle=None;self.reused=False;self.audit={}

    def begin(self, model, optimizer, initial_state_hash):
        self.initial_hash=initial_state_hash
        have_state=self.state_path.exists();have_meta=self.meta_path.exists()
        if have_state!=have_meta:
            raise RuntimeError('Partial shared transition. Preserve and move this folder before retry: '+str(self.folder))
        if not have_state:
            print('COMMON PREFIX: create once -> '+str(self.folder),flush=True)
            return None
        meta=json.loads(self.meta_path.read_text())
        if meta['signature']!=self.signature:
            raise RuntimeError('Shared transition signature changed; use a NEW output root')
        if file_sha(self.state_path)!=meta['file_sha256']:
            raise RuntimeError('Shared transition checksum changed')
        b=torch.load(self.state_path,map_location='cpu',weights_only=True)
        if b['initial_state_sha256']!=initial_state_hash:
            raise RuntimeError('Initial model differs from shared transition source')
        for name in ('model','optimizer','rng','warm_state'):
            if tree_hash(b[name])!=meta['content_hashes'][name]:
                raise RuntimeError('Shared state content differs: '+name)
        self.bundle=b;self.reused=True
        print('COMMON PREFIX: reuse saved model + Adam + RNG; no warmup/transition retraining',flush=True)
        return b

    def finish(self, model, optimizer, generators, warm_state, hist, logs, gradlogs,
               transition_points_hash, source_warm_seconds, source_transition_seconds):
        if self.bundle is None:
            b=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
                   rng=cpu_tree(capture_rng(self.device,generators)),warm_state=cpu_tree(warm_state),
                   initial_state_sha256=self.initial_hash,
                   transition_points_sha256=transition_points_hash,
                   history=copy.deepcopy(hist),objective_logs=copy.deepcopy(logs),
                   gradient_logs=copy.deepcopy(gradlogs),
                   source_warmup_seconds=float(source_warm_seconds),
                   source_transition_seconds=float(source_transition_seconds),
                   creator_objective=self.creator_objective)
            hashes={name:tree_hash(b[name]) for name in ('model','optimizer','rng','warm_state')}
            self.folder.mkdir(parents=True,exist_ok=True)
            if self.state_path.exists() or self.meta_path.exists():
                raise RuntimeError('Shared state already appeared; concurrent writers are not supported')
            temp=self.folder/('common_transition.pt.tmp.'+str(os.getpid()))
            torch.save(b,temp);temp.replace(self.state_path)
            meta=dict(fix_version=FIX_VERSION,signature=self.signature,
                      file_sha256=file_sha(self.state_path),content_hashes=hashes)
            temp_meta=self.folder/('manifest.json.tmp.'+str(os.getpid()))
            temp_meta.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding='utf-8')
            temp_meta.replace(self.meta_path)
            self.bundle=b
        b=self.bundle
        if transition_points_hash!=b['transition_points_sha256']:
            raise RuntimeError('Transition replay points differ from the common prefix')
        # Apply the SAME save/load path to the creator and every subsequent branch.
        model.load_state_dict(b['model'],strict=True)
        optimizer.load_state_dict(copy.deepcopy(b['optimizer']))
        optimizer.zero_grad(set_to_none=True)
        restore_rng(b['rng'],self.device,generators)
        checks={
            'model_exact':tree_hash(model.state_dict())==tree_hash(b['model']),
            'optimizer_exact':tree_hash(optimizer.state_dict())==tree_hash(b['optimizer']),
            'rng_exact':tree_hash(capture_rng(self.device,generators))==tree_hash(b['rng']),
            'warm_state_exact':tree_hash(warm_state)==tree_hash(b['warm_state']),
        }
        if not all(checks.values()):
            raise RuntimeError('Exact shared-start restoration failed: '+str(checks))
        self.audit=dict(fix_version=FIX_VERSION,shared_checkpoint_sha256=file_sha(self.state_path),
            shared_transition_reused=self.reused,creator_objective=b['creator_objective'],
            shared_model_sha256=tree_hash(b['model']),
            transition_optimizer_sha256=tree_hash(b['optimizer']),
            transition_rng_sha256=tree_hash(b['rng']),
            transition_points_sha256=transition_points_hash,exact_checks=checks,
            training_prefix_policy='one common constraint/ramp path, then exact model+Adam+RNG branches',
            runtime_prefix_cached=self.reused,
            prefix_source_warmup_seconds=b['source_warmup_seconds'],
            prefix_source_transition_seconds=b['source_transition_seconds'])
        print('COMMON JOINT START: model=True Adam=True RNG=True',flush=True)
        return self.audit
