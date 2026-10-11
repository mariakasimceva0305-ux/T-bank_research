import pytest
from scripts.analyze import group, paired_delta

def row(reader=0,rule='similarity',seed=0,value=.2):
    return dict(reader_seed=reader,rule=rule,seed=seed,acc_tail=value,stage='main',rho=0.,k=4,capacity=128,steps=400)

def test_mixed_readers_rejected():
    with pytest.raises(ValueError): group([row(),row(reader=1)])
    with pytest.raises(ValueError): paired_delta([row(),row(reader=1)],'x','similarity',0.,4)

def test_unmatched_pairs_rejected():
    with pytest.raises(ValueError): paired_delta([row(),row(rule='x',seed=1)],'x','similarity',0.,4)

def test_paired_difference_preserves_seed_identity():
    rows=[row(seed=0,value=.1),row(seed=1,value=.4),row(rule='x',seed=1,value=.6),row(rule='x',seed=0,value=.2)]
    mean,half,p=paired_delta(rows,'x','similarity',0.,4)
    assert mean==pytest.approx(.15)
    assert half>0
