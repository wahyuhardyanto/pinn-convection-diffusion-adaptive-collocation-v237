#!/usr/bin/env python3
from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
import torch
from unified_context_v237 import UnifiedObjectiveContext
import convection_diffusion_objective_v235 as replay

def main():
    cfg,_,_=replay.configuration('smoke'); cfg=dict(cfg); cfg['n_collocation']=[48]
    ctx=UnifiedObjectiveContext(cfg,gradient_audit=False)
    rng=np.random.default_rng(123)
    p,_=ctx.points(48,'mc',cfg,rng,rng,rng,None,None,'transition',0,False)
    assert p.shape==(48,2)
    assert np.allclose(ctx.pdf_history[0],1) and np.allclose(ctx.weight_history[0],1)
    # Reset contexts to test uniform joint samplers independently.
    results={}
    for m in ('mc','lhs'):
        c=UnifiedObjectiveContext(cfg,gradient_audit=False)
        r0=np.random.default_rng(1); r1=np.random.default_rng(2); rf=np.random.default_rng(3)
        c.points(48,m,cfg,r0,r1,rf,None,None,'transition',0,False)
        pts,_=c.points(48,m,cfg,r0,r1,rf,None,None,'joint',1,False)
        assert pts.shape==(48,2) and np.allclose(c.pdf_history[-1],1) and np.allclose(c.weight_history[-1],1)
        results[m]={'weights_all_one':True,'points_sha256':c.sequence[-1]['points_sha256']}
    print(json.dumps({'version':'2.3.7-unified-four-sampler1','unit_checks':results,'passed':True},indent=2))

if __name__=='__main__': main()
