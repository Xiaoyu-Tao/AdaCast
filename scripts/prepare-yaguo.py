"""Audit the exact prompt_yaguo inputs and derive training-only context selection."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'experiments/yaguo-userprior'
prompt=ROOT/'dataset/prompt_yaguo.md'
paths=[Path(p.replace('\\','/')) for p in re.findall(r'"([^"\n]+\.csv)"',prompt.read_text())]
assert len(paths)==2 and all(p.is_file() for p in paths)
truth,prior=paths
columns=pd.read_csv(truth,nrows=0).columns.tolist()
assert pd.read_csv(prior,nrows=0).columns.tolist()==columns
cities=list(dict.fromkeys(c.split('_')[0] for c in columns if c.endswith('_t2m')))
weather=[f'{city}_{variable}' for city in cities for variable in ['t2m','q','ssrd']]
frames=[]
row_count=0
train_boundary=pd.Timestamp('2025-06-30 09:30')
last=None
for real,pred in zip(pd.read_csv(truth,chunksize=12000,dtype=str,keep_default_na=False),pd.read_csv(prior,chunksize=12000,dtype=str,keep_default_na=False),strict=True):
    assert real.shape==pred.shape
    assert real.drop(columns='OT').equals(pred.drop(columns='OT')), 'Prior changed non-target fields'
    timestamps=pd.to_datetime(real.date,format='mixed')
    assert timestamps.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
    if last is not None: assert timestamps.iloc[0]-last==pd.Timedelta(minutes=15)
    last=timestamps.iloc[-1]
    target=pd.to_numeric(real.OT);prediction=pd.to_numeric(pred.OT)
    assert np.isfinite(target).all() and np.isfinite(prediction).all()
    training=timestamps<=train_boundary
    np.testing.assert_array_equal(target[training],prediction[training])
    frame=real[['date','OT',*weather]].copy()
    frame['prior']=prediction
    frame['date']=timestamps
    for col in ['OT',*weather]: frame[col]=pd.to_numeric(frame[col])
    frames.append(frame)
    row_count+=len(real)
    print(json.dumps({'audited_rows':row_count}),flush=True)
frame=pd.concat(frames,ignore_index=True)
train=frame.loc[frame.date<=train_boundary]
scores=[]
for city in cities:
    correlations={v:float(train.OT.corr(train[f'{city}_{v}'])) for v in ['t2m','q','ssrd']}
    scores.append({'city':city,'correlations':correlations,'score':float(np.mean(np.abs(list(correlations.values()))))})
scores.sort(key=lambda x:-x['score'])
city=scores[0]['city']
# Preserve the provided global event/context columns, including names containing underscores.
global_columns=[c for c in columns if c not in ['date','OT'] and not any(c.startswith(name+'_') for name in cities)]
selected=[f'{city}_{v}' for v in ['t2m','q','ssrd']]+global_columns
assert len(selected)==len(set(selected))
changed=~np.isclose(frame.OT,frame.prior,rtol=0,atol=1e-9)
boundaries={stamp:int(np.flatnonzero(frame.date.eq(pd.Timestamp(stamp)))[0])+1 for stamp in ['2025-06-30 09:30','2025-09-30 09:30']}
train_end,val_end=boundaries.values();fit_end=train_end+(val_end-train_end)//2
anchors={label:int(np.flatnonzero(changed & (frame.index>=start))[0]) for label,start in [('global',train_end),('test_search',val_end)]}
declared_anchor=int(np.flatnonzero(frame.date.eq(pd.Timestamp('2025-07-01 00:00')))[0])
import sys
sys.path.insert(0,str(ROOT/'python/src'))
from adacast_lite_ml.engine.prediction_prior import prediction_window_starts
tasks=[]
for split in ['adjustment_fit','val','test']:
    for w,start in enumerate(prediction_window_starts(split,train_end=train_end,adjustment_fit_end=fit_end,val_end=val_end,row_count=len(frame),seq_len=711,forecast_gap=57,pred_len=96,window_stride=96,prediction_target_anchor=declared_anchor)):
        origin=start+711;label_start=origin+57
        tasks.append({'split':split,'window_index':w,'history_start':str(frame.date.iloc[start]),'history_end':str(frame.date.iloc[origin-1]),'origin_index':origin,'target_start_index':label_start,'target_start':str(frame.date.iloc[label_start]),'target_end':str(frame.date.iloc[label_start+95])})
pd.DataFrame(tasks).to_csv(OUT/'expected-tasks.csv',index=False)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
result={'source_prompt':str(prompt),'truth_path':str(truth),'prior_path':str(prior),'truth_sha256':sha(truth),'prior_sha256':sha(prior),
        'rows':len(frame),'columns':columns,'start':str(frame.date.iloc[0]),'end':str(frame.date.iloc[-1]),'train_end':train_end,'adjustment_fit_end':fit_end,'validation_end':val_end,
        'selected_city':city,'city_selection':{'scope':'training_only','method':'mean absolute Pearson correlation of OT with city t2m, q, ssrd; descriptive proxy, not causal importance','ranking':scores},
        'exogenous_columns':selected,'exogenous_history_steps':288,'target_history_steps':711,'anchors':{label:{'row':index,'timestamp':str(frame.date.iloc[index])} for label,index in anchors.items()},
        'prediction_grid_anchor':{'row':declared_anchor,'timestamp':str(frame.date.iloc[declared_anchor]),'source':'explicit experiment first validation target'},
        'splits':{split:{'tasks':sum(t['split']==split for t in tasks),'first':next(t for t in tasks if t['split']==split),'last':next(t for t in reversed(tasks) if t['split']==split)} for split in ['adjustment_fit','val','test']},
        'checks':{'same_schema_and_timestamps':True,'all_non_target_values_identical':True,'training_target_unchanged':True,'strict_15min_grid':True,'finite_targets':True}}
(OUT/'input-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['columns','city_selection']},ensure_ascii=False,indent=2))
