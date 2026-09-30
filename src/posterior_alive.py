import json, pickle, sys
import numpy as np, torch
from TRL.networks.transition import Network as TN
from TRL.networks.terminal import Network as EN
from TRL.bayes.neural_linear_head import NeuralLinearHead
from TRL.tuning.common import coerce_numeric
N=5
def load(run,ckpt):
    cfg=coerce_numeric(json.load(open(f"{run}/hyper_parameters.txt")))
    d=f"{run}/checkpoints/{ckpt}"; ctx=cfg["transition"]["context_length"]
    tn=TN(cfg["transition"],25,[0,1],p_seq=cfg["transition"]["p_seq"],p_attn=cfg["transition"]["p_attn"],max_steps=cfg["experiment"]["time_limit"],device="cpu")
    tn.load_state_dict(torch.load(f"{d}/transition.pt",map_location="cpu"));tn.eval();tn.unlock_all_dropouts()
    en=EN(25,cfg["terminal"],"cpu");en.load_state_dict(torch.load(f"{d}/terminal.pt",map_location="cpu"));en.eval()
    eps=pickle.load(open(f"{d}/replay.pt","rb"))
    for ep in eps:
        for k in ep:
            if torch.is_tensor(ep[k]): ep[k]=ep[k].cpu()
    return cfg,ctx,tn,en,eps
def probe(run,ckpt):
    cfg,ctx,tn,en,eps=load(run,ckpt)
    head=NeuralLinearHead(cfg["algorithm"],tn,en,25,[0,1],feat_dim=cfg["transition"]["hidden_dim"],device="cpu")
    class DS:pass
    ds=DS();ds.episodes=eps;ds.goal_episodes=[]
    head.update_posteriors(ds)
    # build canonical diagonal probes at each depth: state=r*5+r, action=1 (right), ctx = prior diagonal
    depths=range(5)
    rows=[]
    for r in depths:
        s=r*N+r
        o=torch.zeros(1,ctx,25); o[0,-1,s]=1
        if ctx>=2 and r>0: o[0,-2,(r-1)*N+(r-1)]=1
        t=torch.tensor([[max(r-1,0)]*(ctx-1)+[r]])
        a=torch.tensor([[1]])
        # K sampled predicted next-states -> disagreement = how many distinct argmax cells
        preds=[]
        for _ in range(40):
            head.sample()
            st,_,_=head.predict(t,o,a)
            preds.append(int(st.argmax(-1)))
        distinct=len(set(preds))
        # modal fraction
        vals,cnts=np.unique(preds,return_counts=True)
        modal=cnts.max()/len(preds)
        rows.append((r,distinct,modal))
    return rows
run=sys.argv[1]; ckpt=sys.argv[2]
print(f"{run.split('/')[-2]}: depth | #distinct-next-states(K=40) | modal-frac  (low modal/high distinct = posterior ALIVE, drives exploration)")
for r,d,m in probe(run,ckpt):
    print(f"   depth {r} (diag): {d:2d} distinct | modal {m:.2f}")
