import json,time
from pathlib import Path
import torch,numpy as np
from dam.train import load_reader
from dam.evaluate import run_stream
root=Path.cwd();out=root/'results'/'replication_2026_10_10';out.mkdir(exist_ok=True)
torch.set_num_threads(4)
z=np.load(root/'results/embeddings.npz'); table={t:z['vecs'][i] for i,t in enumerate(z['texts'])}
history=[json.loads(s) for s in (root/'results/runs.jsonl').read_text().splitlines() if s]
results=[]
for reader_seed in range(3):
    ck=root/'checkpoints'/('reader.pt' if reader_seed==0 else f'reader_s{reader_seed}.pt')
    reader,device=load_reader(str(ck))
    for rule in ['similarity','influence_ig_floor']:
        res=run_stream(rule,reader,device,table,rho=0.,seed=0,steps=400,k=4,capacity=128)
        row=res.summary(); row['reader_seed']=reader_seed
        original=next(r for r in history if r.get('reader_seed',0)==reader_seed and r['stage']=='main' and r['rule']==rule and r['rho']==0 and r['seed']==0 and r['k']==4)
        row['previous_acc_tail']=original['acc_tail'];row['absolute_difference']=abs(row['acc_tail']-original['acc_tail'])
        results.append(row)
        (out/'replication.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(row),flush=True)
print('DONE',flush=True)
