"""Point-estimate comparison and integrity audit; no statistical tests."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;BASE=HERE/'results/seed42';SUMMARY=BASE/'summary';SUMMARY.mkdir(parents=True,exist_ok=True)
TARGETS=['total_sugar','reducing_sugar','nicotine','total_nitrogen','potassium']
def met(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);d=p-y
 return {'MAE':float(np.mean(np.abs(d))),'RMSE':float(np.sqrt(np.mean(d*d))),'MAPE':float(np.mean(np.abs(d)/np.maximum(np.abs(y),1e-8))*100),'R2':float(1-np.sum(d*d)/np.sum((y-y.mean())**2))}
def main():
 split=json.loads((ROOT/'data/processed/split_indices.json').read_text());idx=np.asarray(split['test_idx'],int);y=np.load(ROOT/'data/processed/y.npy')[idx].astype(float);grades=np.load(ROOT/'data/processed/grade_ids.npy')[idx]
 mtfile=ROOT/'results/tables/stage5_grade_tsmixer_wind7_default_seed42_predictions.csv';mt=pd.read_csv(mtfile);mtpred=mt.pivot(index='sample_index',columns='target_name',values='y_pred').reindex(index=range(420),columns=TARGETS).to_numpy(float)
 wide=pd.DataFrame({'sample_index':range(420),'original_index':idx,'grade_id':grades,'seed':42});rows=[];audit=[];times=[];epochs=[]
 for j,t in enumerate(TARGETS):
  tag=f'grade_tsmixer_wind7_single_{t}_seed42';d=BASE/t;tab=d/'tables';w=pd.read_csv(tab/f'{tag}_predictions_wide.csv').sort_values('sample_index');m=json.loads((tab/f'{tag}_run_metadata.json').read_text(encoding='utf-8'));pred=w[f'pred_{t}'].to_numpy(float)
  sm=met(y[:,j],pred);mm=met(y[:,j],mtpred[:,j]);row={'target_index':j,'target_name':t}
  for k in ('MAE','RMSE','MAPE','R2'):row[f'multi_{k.lower()}']=mm[k];row[f'single_{k.lower()}']=sm[k];row[f'delta_single_minus_multi_{k.lower()}']=sm[k]-mm[k]
  rows.append(row);wide[f'true_{t}']=y[:,j];wide[f'pred_{t}']=pred;times.append(m['training_time_seconds']);epochs.append(m['stop_epoch'])
  saved=pd.read_csv(tab/f'{tag}_metrics.csv').iloc[0]
  audit.append({'target_name':t,'parameter_count':m['parameter_count'],'best_epoch':m['best_epoch'],'stop_epoch':m['stop_epoch'],'training_time_seconds':m['training_time_seconds'],'wide_rows':len(w),'original_index_exact':bool(np.array_equal(w.original_index.to_numpy(int),idx)),'grade_id_exact':bool(np.array_equal(w.grade_id.to_numpy(int),grades)),'truth_matches_y_npy_at_csv_precision':bool(np.allclose(w[f'true_{t}'].to_numpy(float),y[:,j],rtol=0,atol=1e-12)),'finite_predictions':bool(np.isfinite(pred).all()),'saved_metrics_reproduced_within_csv_tolerance':bool(all(np.isclose(float(saved[k]),sm[k],rtol=0,atol=1e-6) for k in ('MAE','RMSE','MAPE','R2'))),'max_abs_saved_metric_recompute_difference':float(max(abs(float(saved[k])-sm[k]) for k in ('MAE','RMSE','MAPE','R2'))),'checkpoint_exists':(d/'checkpoints'/f'{tag}_best.pth').exists(),'history_exists':(tab/f'{tag}_training_history.csv').exists(),'log_exists':(d/'logs'/f'{tag}.log').exists(),'curve_exists':(d/'figures'/f'{tag}_training_curve.png').exists()})
 comp=pd.DataFrame(rows);au=pd.DataFrame(audit)
 st={k:float(comp[f'single_{k.lower()}'].mean()) for k in ('MAE','RMSE','MAPE','R2')};mtavg={k:float(comp[f'multi_{k.lower()}'].mean()) for k in ('MAE','RMSE','MAPE','R2')}
 overall=pd.DataFrame([{'setting':'Multi-target','avg_mae':mtavg['MAE'],'avg_rmse':mtavg['RMSE'],'avg_mape':mtavg['MAPE'],'avg_r2':mtavg['R2']},{'setting':'Five Single-target models','avg_mae':st['MAE'],'avg_rmse':st['RMSE'],'avg_mape':st['MAPE'],'avg_r2':st['R2']}]);overall.loc[1,'delta_avg_mape_st_minus_mt']=st['MAPE']-mtavg['MAPE']
 params=pd.DataFrame([{'setting':'Multi-target','models_required':1,'parameters_per_model':179093,'total_parameters':179093},{'setting':'Five Single-target models','models_required':5,'parameters_per_model':int(au.parameter_count.iloc[0]),'total_parameters':int(au.parameter_count.sum())}])
 cost=au[['target_name','best_epoch','stop_epoch','training_time_seconds']].copy();cost.loc[len(cost)]={'target_name':'TOTAL','best_epoch':int(au.best_epoch.sum()),'stop_epoch':int(au.stop_epoch.sum()),'training_time_seconds':float(sum(times))}
 comp.to_csv(SUMMARY/'single_vs_multi_per_target.csv',index=False,encoding='utf-8-sig');overall.to_csv(SUMMARY/'single_vs_multi_overall.csv',index=False,encoding='utf-8-sig');params.to_csv(SUMMARY/'parameter_comparison.csv',index=False,encoding='utf-8-sig');cost.to_csv(SUMMARY/'single_target_training_cost.csv',index=False,encoding='utf-8-sig');wide.to_csv(SUMMARY/'single_target_merged_predictions_wide.csv',index=False,encoding='utf-8-sig');au.to_csv(SUMMARY/'single_target_completeness_audit.csv',index=False,encoding='utf-8-sig')
 boolean_checks=[c for c in au.columns if au[c].dtype==bool]
 payload={'analysis_type':'point estimates only; no statistical tests','multi_target_prediction_source':str(mtfile),'truth_source':str(ROOT/'data/processed/y.npy'),'test_records':420,'per_target':comp.to_dict('records'),'overall':overall.to_dict('records'),'parameters':params.to_dict('records'),'training_cost':cost.to_dict('records'),'all_integrity_checks_pass':bool(au[boolean_checks].to_numpy().all()),'cluster_statistics_ready':True,'cluster_statistics_note':'original_index aligns all 420 records; join sample_info.csv place/year for county-year clusters; not run in this stage'}
 (SUMMARY/'single_vs_multi_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
 print(comp.to_string(index=False));print('\n',overall.to_string(index=False));print('\n',params.to_string(index=False));print('\n',cost.to_string(index=False));print('\nAUDIT',payload['all_integrity_checks_pass'])
if __name__=='__main__':main()
