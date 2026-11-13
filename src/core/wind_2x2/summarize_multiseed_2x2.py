"""Summarize the fixed 10-seed paired Wind 2x2 stability experiment."""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
OUT=HERE/'multiseed'; SUMMARY=OUT/'summary'; SUMMARY.mkdir(parents=True,exist_ok=True)
SEEDS=[42,123,456,789,2024,3407,5678,7777,8888,9999]
CONFIGS=['O5','O6','W6','W7']; TARGETS=['total_sugar','reducing_sugar','nicotine','total_nitrogen','potassium']

def avg_mape_long(path,y_true):
    d=pd.read_csv(path); pred=d.pivot(index='sample_index',columns='target_name',values='y_pred').reindex(index=range(420),columns=TARGETS).to_numpy()
    return float(np.mean(np.abs(pred-y_true)/np.maximum(np.abs(y_true),1e-8)*100))
def avg_mape_wide(path,y_true):
    d=pd.read_csv(path).sort_values('sample_index'); pred=d[[f'pred_{t}' for t in TARGETS]].to_numpy()
    return float(np.mean(np.abs(pred-y_true)/np.maximum(np.abs(y_true),1e-8)*100))
def avg_mape_metrics(path):
    d=pd.read_csv(path); return float(d.loc[d.metric=='MAPE','Avg'].iloc[0])
def holm(x):
    p=np.asarray(x,float); o=np.argsort(p); s=p[o]; a=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*s)); z=np.empty_like(a); z[o]=a; return z
def bootstrap_ci(x,n=100_000):
    rng=np.random.default_rng(42); draws=rng.integers(0,len(x),size=(n,len(x)))
    return np.quantile(np.asarray(x)[draws].mean(1),[.025,.975])
def exact_signflip(x):
    x=np.asarray(x,float); obs=abs(x.mean()); vals=[]
    for signs in itertools.product((-1.,1.),repeat=len(x)): vals.append(abs(np.mean(x*np.asarray(signs))))
    return float(np.mean(np.asarray(vals)>=obs-1e-15))
def describe(name,x,kind):
    x=np.asarray(x,float); q1,med,q3=np.quantile(x,[.25,.5,.75]); lo,hi=bootstrap_ci(x)
    return {'name':name,'kind':kind,'n_seeds':len(x),'mean':x.mean(),'sample_sd':x.std(ddof=1),
            'median':med,'q1':q1,'q3':q3,'iqr':q3-q1,'min':x.min(),'max':x.max(),
            'bootstrap_mean_ci95_low':lo,'bootstrap_mean_ci95_high':hi,
            'n_negative':int((x<0).sum()),'n_positive':int((x>0).sum()),'n_zero':int((x==0).sum()),
            'majority_direction_consistency':max((x<0).sum(),(x>0).sum())/len(x)}

def main():
    rows=[]; checks=[]
    split=json.loads((ROOT/'data/processed/split_indices.json').read_text(encoding='utf-8'))
    test_idx=np.asarray(split['test_idx'],dtype=int); y_true=np.load(ROOT/'data/processed/y.npy')[test_idx].astype(float)
    existing={
      'O5':HERE/'results/original5/seed42/tables/grade_tsmixer_original5_default_seed42_predictions_wide.csv',
      'O6':ROOT/'results/tables/stage5_grade_tsmixer_original6_default_seed42_predictions.csv',
      'W6':ROOT/'results/tables/stage5_grade_tsmixer_wind6_default_seed42_predictions.csv',
      'W7':ROOT/'results/tables/stage5_grade_tsmixer_wind7_default_seed42_predictions.csv'}
    for seed in SEEDS:
      row={'seed':seed}
      for c in CONFIGS:
        if seed==42: val=avg_mape_wide(existing[c],y_true) if c=='O5' else avg_mape_long(existing[c],y_true); source=existing[c]
        else:
          base=OUT/f'seed{seed}'/c.lower(); tag=f'grade_tsmixer_{c.lower()}_default_seed{seed}'
          source=base/'tables'/f'{tag}_metrics.csv'
          files=[source,base/'tables'/f'{tag}_predictions_wide.csv',base/'tables'/f'{tag}_training_history.csv',
                 base/'tables'/f'{tag}_run_metadata.json',base/'checkpoints'/f'{tag}_best.pth',base/'logs'/f'{tag}.log']
          val=avg_mape_wide(files[1],y_true)
          wide=pd.read_csv(files[1]); hist=pd.read_csv(files[2]); meta=json.loads(files[3].read_text(encoding='utf-8'))
          checks.append({'seed':seed,'configuration':c,'all_required_files_exist_nonempty':all(p.exists() and p.stat().st_size>0 for p in files),
                         'wide_420_rows':len(wide)==420,'original_index_unique':wide.original_index.nunique()==420,
                         'history_matches_stop_epoch':len(hist)==meta['stop_epoch'],'best_epoch_valid':1<=meta['best_epoch']<=meta['stop_epoch'],
                         'finite_history':bool(np.isfinite(hist[['train_loss','val_loss']].to_numpy()).all()),
                         'avg_mape':val,'best_epoch':meta['best_epoch'],'stop_epoch':meta['stop_epoch'],'duration_seconds':meta['duration_seconds']})
        row[c]=val
      rows.append(row)
    m=pd.DataFrame(rows)
    e=pd.DataFrame({'seed':m.seed,'O6_minus_O5':m.O6-m.O5,'W6_minus_O5':m.W6-m.O5,
                    'W7_minus_O6':m.W7-m.O6,'W7_minus_W6':m.W7-m.W6,'W7_minus_O5':m.W7-m.O5})
    e['Interaction']=m.W7-m.W6-m.O6+m.O5
    summaries=[describe(c,m[c],'configuration_avg_mape') for c in CONFIGS]
    effect_rows=[]
    for c in e.columns[1:]:
      x=e[c].to_numpy(); d=describe(c,x,'effect_delta_mape_pp')
      t=stats.ttest_1samp(x,0); w=stats.wilcoxon(x,zero_method='wilcox',alternative='two-sided',method='auto')
      d.update({'exact_signflip_p_raw':exact_signflip(x),'one_sample_t_statistic':t.statistic,'one_sample_t_p_raw':t.pvalue,
                'wilcoxon_statistic':w.statistic,'wilcoxon_p_raw':w.pvalue})
      effect_rows.append(d)
    es=pd.DataFrame(effect_rows)
    for raw,adj in [('exact_signflip_p_raw','exact_signflip_p_holm'),('one_sample_t_p_raw','one_sample_t_p_holm'),('wilcoxon_p_raw','wilcoxon_p_holm')]: es[adj]=holm(es[raw])
    summary=pd.DataFrame(summaries)
    check=pd.DataFrame(checks)
    # Flag descriptive 1.5-IQR outliers only; never exclude them.
    flags=[]
    for c in CONFIGS:
      q1,q3=np.quantile(m[c],[.25,.75]); i=q3-q1
      for seed,val in zip(m.seed,m[c]):
        if val<q1-1.5*i or val>q3+1.5*i: flags.append({'seed':int(seed),'configuration':c,'avg_mape':val,'rule':'outside 1.5 IQR; retained'})
    m.to_csv(SUMMARY/'ten_seed_four_configuration_avg_mape.csv',index=False,encoding='utf-8-sig')
    e.to_csv(SUMMARY/'ten_seed_six_effects.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(SUMMARY/'configuration_cross_seed_summary.csv',index=False,encoding='utf-8-sig')
    es.to_csv(SUMMARY/'effect_cross_seed_statistics.csv',index=False,encoding='utf-8-sig')
    check.to_csv(SUMMARY/'new_run_completeness_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(flags).to_csv(SUMMARY/'descriptive_outlier_flags.csv',index=False,encoding='utf-8-sig')
    payload={'seed_set':SEEDS,'seed42_reused_not_retrained':True,'new_training_runs':36,'failed_runs':0,
             'delta_convention':'after minus before; negative is improvement','bootstrap_iterations':100000,
             'exact_signflip_enumerations':1024,'multiplicity':'Holm across all six effects, separately for sign-flip, t, and Wilcoxon p-values',
             'configurations':summary.to_dict('records'),'effects':es.to_dict('records'),'outlier_flags':flags,
             'completeness_all_pass':bool(check.iloc[:,2:7].to_numpy().all())}
    (SUMMARY/'ten_seed_2x2_stability_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n10x4\n',m.to_string(index=False)); print('\nEFFECTS\n',e.to_string(index=False)); print('\nSUMMARY\n',summary.to_string(index=False)); print('\nSTATS\n',es.to_string(index=False)); print('\nFLAGS',flags); print('ALL CHECKS',payload['completeness_all_pass'])
if __name__=='__main__': main()
