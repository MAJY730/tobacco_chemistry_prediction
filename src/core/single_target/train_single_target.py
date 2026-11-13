"""Train five independent single-target Grade-TSMixer Wind-7 models."""
from __future__ import annotations
import hashlib,json,pickle,platform,sys,time,traceback
from datetime import datetime
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader,TensorDataset

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(ROOT))
from models.top_ts_models import build_model
TARGETS=['total_sugar','reducing_sugar','nicotine','total_nitrogen','potassium']
DATA=ROOT/'data/processed'; STD=DATA/'standardized'; OUT=HERE/'results/seed42'
SEED=42; BATCH=32; MAX_EPOCHS=300; PATIENCE=30; LR=1e-3; WD=1e-4

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def set_seed():
 torch.set_num_threads(8); np.random.seed(SEED); torch.manual_seed(SEED)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);d=p-y
 return {'MAE':float(np.mean(np.abs(d))),'RMSE':float(np.sqrt(np.mean(d*d))),
         'MAPE':float(np.mean(np.abs(d)/np.maximum(np.abs(y),1e-8))*100),
         'R2':float(1-np.sum(d*d)/np.sum((y-y.mean())**2))}
def complete(tdir,tag):
 files=[tdir/f'{tag}_metrics.csv',tdir/f'{tag}_predictions_wide.csv',tdir/f'{tag}_training_history.csv',tdir/f'{tag}_run_metadata.json']
 try:return all(x.exists() and x.stat().st_size for x in files) and len(pd.read_csv(files[1]))==420
 except:return False
def run(target_index,common):
 target=TARGETS[target_index]; base=OUT/target; tables=base/'tables'; ckpts=base/'checkpoints';logs=base/'logs';figs=base/'figures'
 for x in (tables,ckpts,logs,figs):x.mkdir(parents=True,exist_ok=True)
 tag=f'grade_tsmixer_wind7_single_{target}_seed42'
 if complete(tables,tag): print('SKIP',target,flush=True);return
 logfile=logs/f'{tag}.log';logfile.write_text('',encoding='utf-8')
 def log(s):
  z=f'{datetime.now().astimezone().isoformat()} {s}';print(z,flush=True)
  with logfile.open('a',encoding='utf-8') as f:f.write(z+'\n')
 set_seed(); start=datetime.now().astimezone();tic=time.perf_counter()
 X,y,g=common['X'],common['ys'][:,target_index:target_index+1],common['g'];tr,va,te=common['tr'],common['va'],common['te']
 train=TensorDataset(torch.tensor(X[tr],dtype=torch.float32),torch.tensor(y[tr],dtype=torch.float32),torch.tensor(g[tr],dtype=torch.long))
 val=TensorDataset(torch.tensor(X[va],dtype=torch.float32),torch.tensor(y[va],dtype=torch.float32),torch.tensor(g[va],dtype=torch.long))
 test=TensorDataset(torch.tensor(X[te],dtype=torch.float32),torch.tensor(g[te],dtype=torch.long))
 tl=DataLoader(train,batch_size=BATCH,shuffle=True);vl=DataLoader(val,batch_size=BATCH);xel=DataLoader(test,batch_size=BATCH)
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model=build_model('grade_tsmixer',7,153,'default',16,5);model.fc2=nn.Linear(64,1);model=model.to(dev)
 params=sum(p.numel() for p in model.parameters() if p.requires_grad)
 if params!=178833:raise ValueError(f'parameter count {params}')
 opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD);sched=ReduceLROnPlateau(opt,mode='min',factor=.5,patience=10);crit=nn.MSELoss()
 best=float('inf');state=None;be=0;wait=0;hist=[];log(f'START target={target} index={target_index} params={params} device={dev}')
 for epoch in range(1,MAX_EPOCHS+1):
  model.train();s=0.
  for bx,by,bg in tl:
   bx,by,bg=bx.to(dev),by.to(dev),bg.to(dev);opt.zero_grad();loss=crit(model(bx,bg),by);loss.backward();opt.step();s+=loss.item()*len(bx)
  trainloss=s/len(train);model.eval();s=0.
  with torch.no_grad():
   for bx,by,bg in vl:
    bx,by,bg=bx.to(dev),by.to(dev),bg.to(dev);s+=crit(model(bx,bg),by).item()*len(bx)
  valloss=s/len(val);sched.step(valloss);hist.append({'epoch':epoch,'train_loss':trainloss,'val_loss':valloss,'learning_rate':opt.param_groups[0]['lr']})
  if valloss<best:best=valloss;be=epoch;wait=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:wait+=1
  if epoch==1 or epoch%10==0:log(f'epoch={epoch} train={trainloss:.8f} val={valloss:.8f}')
  if wait>=PATIENCE:break
 stop=epoch
 if state is None:raise RuntimeError('no checkpoint');model.load_state_dict(state)
 # The assignment above is intentionally repeated outside the guard for clarity.
 model.load_state_dict(state);model.eval();ps=[]
 with torch.no_grad():
  for bx,bg in xel:ps.append(model(bx.to(dev),bg.to(dev)).cpu().numpy())
 ps=np.concatenate(ps)[:,0];scale=common['scaler'];pred=ps*scale['std'][target_index]+scale['mean'][target_index];truth=common['yraw'][te,target_index].astype(float)
 met=metrics(truth,pred);pd.DataFrame([{'target_index':target_index,'target_name':target,**met}]).to_csv(tables/f'{tag}_metrics.csv',index=False,encoding='utf-8-sig')
 (tables/f'{tag}_metrics.json').write_text(json.dumps({'target_index':target_index,'target_name':target,**met},indent=2),encoding='utf-8')
 h=pd.DataFrame(hist);h.to_csv(tables/f'{tag}_training_history.csv',index=False,encoding='utf-8-sig')
 wide=pd.DataFrame({'sample_index':np.arange(420),'original_index':te,'grade_id':g[te],'seed':SEED,f'true_{target}':truth,f'pred_{target}':pred})
 wide.to_csv(tables/f'{tag}_predictions_wide.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame({'sample_index':np.arange(420),'original_index':te,'grade_id':g[te],'target_index':target_index,'target_name':target,'seed':SEED,'y_true':truth,'y_pred':pred}).to_csv(tables/f'{tag}_predictions_long.csv',index=False,encoding='utf-8-sig')
 torch.save({'model_state_dict':state,'target_index':target_index,'target_name':target,'seed':SEED,'parameter_count':params,'best_epoch':be,'stop_epoch':stop,'best_validation_mse':best},ckpts/f'{tag}_best.pth')
 fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(h.epoch,h.train_loss,label='Train');ax.plot(h.epoch,h.val_loss,label='Validation');ax.set(xlabel='Epoch',ylabel='MSE',title=f'Single-target {target}');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figs/f'{tag}_training_curve.png',dpi=180);plt.close(fig)
 meta={'target_name':target,'target_index':target_index,'seed':SEED,'parameter_count':params,'best_epoch':be,'stop_epoch':stop,'best_validation_mse':best,'training_time_seconds':time.perf_counter()-tic,'started_at':start.isoformat(),'finished_at':datetime.now().astimezone().isoformat(),'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':str(dev)},'split':{'train':1952,'validation':418,'test':420,'source':str(DATA/'split_indices.json'),'sha256':common['splitsha']},'inputs':{'climate':str(STD/'X_wind7_scaled.npy'),'grade_ids':str(DATA/'grade_ids.npy'),'y_scaled':str(STD/'y_scaled.npy'),'y_raw':str(DATA/'y.npy'),'y_scaler':str(DATA/'scalers/y_scaler.pkl')},'model':{'base':'GradeAwareTSMixer default constructed with output_dim=5','only_replacement':'outer model.fc2 = nn.Linear(64,1)','input_shape':[153,7],'grade_embedding':[16,8],'d_model':64,'blocks':3,'hidden_dim':128,'dropout':.2},'training':{'optimizer':'AdamW','lr':LR,'weight_decay':WD,'batch_size':BATCH,'max_epochs':MAX_EPOCHS,'scheduler':'ReduceLROnPlateau factor=.5 patience=10','early_stopping_patience':PATIENCE,'loss':'MSE'}}
 (tables/f'{tag}_run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');log(f'DONE best={be} stop={stop} MAPE={met["MAPE"]:.8f} seconds={meta["training_time_seconds"]:.2f}')
def main():
 spath=DATA/'split_indices.json';sp=json.loads(spath.read_text());tr=np.array(sp['train_idx']);va=np.array(sp['val_idx']);te=np.array(sp['test_idx'])
 if tuple(map(len,(tr,va,te)))!=(1952,418,420):raise ValueError('split changed')
 with (DATA/'scalers/y_scaler.pkl').open('rb') as f:sc=pickle.load(f)
 common={'X':np.load(STD/'X_wind7_scaled.npy'),'ys':np.load(STD/'y_scaled.npy'),'yraw':np.load(DATA/'y.npy'),'g':np.load(DATA/'grade_ids.npy'),'tr':tr,'va':va,'te':te,'scaler':sc,'splitsha':sha(spath)}
 if common['X'].shape!=(2790,153,7) or common['ys'].shape!=(2790,5):raise ValueError('shape')
 failures=[]
 for i in range(5):
  try:run(i,common)
  except Exception as e:failures.append({'target':TARGETS[i],'error':repr(e),'traceback':traceback.format_exc()});print('FAILED',TARGETS[i],e,flush=True)
 (HERE/'results/seed42/training_failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
 if failures:raise RuntimeError(f'{len(failures)} failures')
 print('ALL 5 SINGLE-TARGET RUNS COMPLETE',flush=True)
if __name__=='__main__':main()
