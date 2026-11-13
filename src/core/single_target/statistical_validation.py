"""County-year cluster validation of Single-target minus Multi-target APE."""
from __future__ import annotations
import json,logging,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,scipy
from scipy import stats

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=HERE/'results/seed42/statistical_validation';OUT.mkdir(parents=True,exist_ok=True)
TARGETS=['total_sugar','reducing_sugar','nicotine','total_nitrogen','potassium']
N_PERM=100_000;N_BOOT=10_000;SEED=42
logging.basicConfig(filename=OUT/'analysis.log',filemode='w',level=logging.INFO,format='%(asctime)s %(message)s')
log=logging.getLogger(__name__)
def holm(x):
 p=np.asarray(x,float);o=np.argsort(p);s=p[o];a=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*s));z=np.empty_like(a);z[o]=a;return z
def analyze(diff,codes):
 clusters=np.unique(codes);members=[np.flatnonzero(codes==c) for c in clusters];sizes=np.array([len(i) for i in members],float);means=np.array([diff[i].mean() for i in members]);point=float(diff.mean())
 rng=np.random.default_rng(SEED);extreme=1
 for start in range(0,N_PERM,10_000):
  n=min(10_000,N_PERM-start);sign=rng.choice(np.array([-1.,1.]),size=(n,len(clusters)));v=(sign@(means*sizes))/sizes.sum();extreme+=int(np.count_nonzero(np.abs(v)>=abs(point)))
 pp=extreme/(N_PERM+1)
 rng=np.random.default_rng(SEED);draw=rng.integers(0,len(clusters),size=(N_BOOT,len(clusters)));boot=(means*sizes)[draw].sum(1)/sizes[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
 t=stats.ttest_1samp(means,0);w=stats.wilcoxon(means,zero_method='wilcox',alternative='two-sided',method='auto')
 return {'delta_mape_pp':point,'n_records':len(diff),'n_county_year_clusters':len(clusters),'permutation_p_raw':float(pp),'bootstrap_ci95_low_pp':float(lo),'bootstrap_ci95_high_pp':float(hi),'bootstrap_ci_crosses_zero':bool(lo<=0<=hi),'group_t_statistic':float(t.statistic),'group_t_p_raw':float(t.pvalue),'group_wilcoxon_statistic':float(w.statistic),'group_wilcoxon_p_raw':float(w.pvalue)},means,sizes,boot
def main():
 split=json.loads((ROOT/'data/processed/split_indices.json').read_text());idx=np.asarray(split['test_idx'],int);y=np.load(ROOT/'data/processed/y.npy')[idx].astype(float);grade=np.load(ROOT/'data/processed/grade_ids.npy')[idx]
 stfile=HERE/'results/seed42/summary/single_target_merged_predictions_wide.csv';st=pd.read_csv(stfile).sort_values('sample_index')
 mtfile=ROOT/'results/tables/stage5_grade_tsmixer_wind7_default_seed42_predictions.csv';mt=pd.read_csv(mtfile);mp=mt.pivot(index='sample_index',columns='target_name',values='y_pred').reindex(index=range(420),columns=TARGETS).to_numpy(float);sp=st[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
 if len(idx)!=420 or len(np.unique(idx))!=420 or not np.array_equal(st.original_index.to_numpy(int),idx) or not np.array_equal(st.grade_id.to_numpy(int),grade):raise ValueError('record alignment failed')
 info=pd.read_csv(ROOT/'data/processed/sample_info.csv').set_index('original_index').loc[idx].reset_index();info['cluster']=info.place.astype(str)+'__'+info.year.astype(str);codes,names=pd.factorize(info.cluster,sort=True)
 if len(names)!=149:raise ValueError(f'clusters={len(names)}')
 ape_st=np.abs(sp-y)/np.maximum(np.abs(y),1e-8)*100;ape_mt=np.abs(mp-y)/np.maximum(np.abs(y),1e-8)*100;diff=ape_st-ape_mt
 overall,om,os,ob=analyze(diff.mean(1),codes);overall['analysis']='overall_mean_of_five_targets'
 target_rows=[];cluster_rows=[];boot_targets={}
 for j,tg in enumerate(TARGETS):
  r,gm,gs,b=analyze(diff[:,j],codes);r.update({'target_index':j,'target_name':tg});target_rows.append(r);boot_targets[tg]={'iterations':N_BOOT,'ci95_low_pp':r['bootstrap_ci95_low_pp'],'ci95_high_pp':r['bootstrap_ci95_high_pp'],'crosses_zero':r['bootstrap_ci_crosses_zero']}
  cluster_rows.extend({'analysis':tg,'cluster':str(n),'n_records':int(sz),'cluster_mean_delta_mape_pp':float(v)} for n,sz,v in zip(names,gs,gm))
 tr=pd.DataFrame(target_rows)
 for raw,adj in [('permutation_p_raw','permutation_p_holm'),('group_t_p_raw','group_t_p_holm'),('group_wilcoxon_p_raw','group_wilcoxon_p_holm')]:tr[adj]=holm(tr[raw])
 overall_df=pd.DataFrame([overall]);overall_df.to_csv(OUT/'overall_cluster_statistics.csv',index=False,encoding='utf-8-sig');tr.to_csv(OUT/'per_target_cluster_statistics.csv',index=False,encoding='utf-8-sig')
 cluster_rows.extend({'analysis':'overall','cluster':str(n),'n_records':int(sz),'cluster_mean_delta_mape_pp':float(v)} for n,sz,v in zip(names,os,om));pd.DataFrame(cluster_rows).to_csv(OUT/'cluster_level_differences.csv',index=False,encoding='utf-8-sig')
 rec=info[['original_index','place','year','cluster']].copy();rec['grade_id']=grade
 for j,tg in enumerate(TARGETS):rec[f'ape_single_{tg}']=ape_st[:,j];rec[f'ape_multi_{tg}']=ape_mt[:,j];rec[f'difference_{tg}']=diff[:,j]
 rec['mean_ape_single']=ape_st.mean(1);rec['mean_ape_multi']=ape_mt.mean(1);rec['difference_overall']=diff.mean(1);rec.to_csv(OUT/'record_level_ape_differences.csv',index=False,encoding='utf-8-sig')
 (OUT/'overall_cluster_bootstrap.json').write_text(json.dumps({'iterations':N_BOOT,'point_estimate_pp':overall['delta_mape_pp'],'ci95_low_pp':overall['bootstrap_ci95_low_pp'],'ci95_high_pp':overall['bootstrap_ci95_high_pp'],'crosses_zero':overall['bootstrap_ci_crosses_zero']},indent=2),encoding='utf-8')
 (OUT/'per_target_cluster_bootstrap.json').write_text(json.dumps(boot_targets,indent=2),encoding='utf-8')
 payload={'method':{'truth_source':str(ROOT/'data/processed/y.npy'),'st_predictions':str(stfile),'mt_predictions':str(mtfile),'difference':'Single-target APE minus Multi-target APE; positive favors Multi-target','permutation_iterations':N_PERM,'bootstrap_iterations':N_BOOT,'seed':SEED,'holm':'five targets separately for each p-value family','python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'integrity':{'test_records':420,'unique_original_index':420,'county_year_clusters':149,'cluster_size_min':int(pd.Series(codes).value_counts().min()),'cluster_size_median':float(pd.Series(codes).value_counts().median()),'cluster_size_max':int(pd.Series(codes).value_counts().max()),'st_mt_record_alignment':True,'truth_source_y_npy':True,'target_order':TARGETS},'overall':overall,'per_target':tr.to_dict('records')}
 (OUT/'statistical_validation_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');log.info('Completed: %s',json.dumps(payload,ensure_ascii=False));print(overall_df.to_string(index=False));print('\n',tr.to_string(index=False));print('\n',payload['integrity'])
if __name__=='__main__':main()
