"""Step 3: descriptive pre-reference gap evidence; always awaits user review."""
# coding: utf-8
from pathlib import Path
import sys,json,hashlib,time,platform
root=Path(__file__).resolve().parents[1]
import config
import os
os.environ.setdefault("MPLCONFIGDIR", str(config.RUN_PROC_DIR / "matplotlib"))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.churn_features import pre_reference_evidence

def main():
    start=time.monotonic()
    def digest(path):
     h=hashlib.sha256()
     with Path(path).open('rb') as f:
      for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
     return h.hexdigest()
    execution=json.loads((config.RUN_REPORTS_DIR/'notebook04_execution.json').read_text())
    assert execution['stage']=='completed' and execution['raw_metadata_unchanged']
    assert not config.RISK_BASIS_PATH.exists(),'Do not overwrite stage-3 evidence'
    path=config.RUN_PROC_DIR/'user_activity_days.parquet'
    print('Reading generated activity-day data, not raw events',flush=True)
    daily=pd.read_parquet(path,columns=['user_id','date'])
    full=json.loads((config.RUN_REPORTS_DIR/'full_period_gap_distribution.json').read_text())
    report=pre_reference_evidence(daily,full)
    print('Pre-reference gap distribution calculated',flush=True)
    ref=pd.Timestamp(config.AB_REFERENCE_DATE,tz='UTC')
    last=daily.loc[daily.date.le(ref)].groupby('user_id')['date'].max()
    report['pre_reference']['observed_users']=len(last)
    recency=(ref-last).dt.days.rename('recency_days')
    labels=pd.read_parquet(config.RUN_PROC_DIR/'churn_labels.parquet',columns=['user_id','status']).set_index('user_id')
    joined=recency.to_frame().join(labels,validate='one_to_one')
    assert joined.status.isin(['retained','churned']).all()
    joined['returned']=joined.status.eq('retained')
    agg=joined.groupby('recency_days').agg(users=('status','size'),returned_users=('returned','sum')).reindex(range(config.MAX_RECENCY_DAYS+1),fill_value=0)
    agg['return_rate']=agg.returned_users.div(agg.users.where(agg.users.ne(0)))
    assert int(agg.users.sum())==len(last)
    assert int(agg.returned_users.sum())==int(labels.status.eq('retained').sum())
    agg_path=config.RUN_REPORTS_DIR/'recency_day_outcome_comparison.csv';agg.to_csv(agg_path,index=True)
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(agg.index,agg.return_rate,linewidth=1.1)
    ax.set(xlabel='Recency at 2020-03-26 (UTC calendar days)',ylabel='Observed return rate: 2020-03-27 to 2020-04-30',title='Descriptive comparison by recency; not an intervention effect')
    ax.set_ylim(0,1);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(config.RUN_FIGURES_DIR/'recency_day_outcome_comparison.png',dpi=150)
    report['descriptive_curve']={'csv':str(agg_path.relative_to(root)),'source':'04 regenerated activity-day last dates joined to new outcome labels; not independently scanned Recency candidate','users_total':int(agg.users.sum()),'returned_users_total':int(agg.returned_users.sum()),'effect_claim':False}
    report['input_files']={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in [path,config.RUN_PROC_DIR/'churn_labels.parquet',config.RUN_REPORTS_DIR/'full_period_gap_distribution.json']}
    report['code_sha256']={str(p.relative_to(root)):digest(p) for p in [root/'config.py',root/'src/churn_features.py',root/'src/load.py']}
    report['environment']={'python':platform.python_version(),'pandas':pd.__version__}
    report['elapsed_seconds']=round(time.monotonic()-start,2)
    report['next_gate']='STOP: explicit user confirmation required before step 4'
    config.RISK_BASIS_PATH.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['full_period','pre_reference']},ensure_ascii=False,indent=2),flush=True)
    print('PRE',json.dumps({k:v for k,v in report['pre_reference'].items() if k!='gap_histogram'}),flush=True)
    print('FULL',json.dumps({k:v for k,v in full.items() if k!='gap_histogram'}),flush=True)


if __name__ == "__main__":
    main()
