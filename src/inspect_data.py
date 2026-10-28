import pandas as pd, numpy as np, sys; sys.path.insert(0,'src'); from utils import parse_ist
R='data/raw/'
tr=pd.read_csv(R+'Train.csv')
tr['ts']=parse_ist(tr.timestamp_ist)
tr['avail']=parse_ist(tr.available_at)
te=pd.read_csv(R+'Test.csv')
print(tr.dtypes); print(tr.shape)
print('dups (loc,ts):',tr.duplicated(['location_id','ts']).sum())
print('pm25!=mean:',(tr.pm25-tr['mean']).abs().gt(1e-9).sum())
print('flag_high:',tr.flag_high_1000.sum(), 'count dist', tr['count'].value_counts().to_dict())
for loc,g in tr.groupby('location_id'):
    g=g.sort_values('ts'); full=pd.date_range(g.ts.min(),g.ts.max(),freq='h')
    lag=(g.avail-g.ts).dt.total_seconds()/3600
    print(loc,g.station_name.iloc[0],'n',len(g),'span',g.ts.min(),'->',g.ts.max(),'coverage %.1f%%'%(100*len(g)/len(full)),
      'pm25 q:',np.round(g.pm25.quantile([0,.01,.5,.99,1]).values,1),'mean %.1f'%g.pm25.mean(),
      'avail lag h med %.0f min %.0f'%(lag.median(),lag.min()))
print('last ts overall',tr.ts.max())
te['end']=parse_ist(te.target_end_ist)
print(te.shape, te.ID.nunique(), te.groupby('location_id').size().to_dict())
print(te.groupby('scoring_block').end.agg(['min','max','count']))
print(te.groupby(['scoring_block',te.end.dt.date]).size().unstack(0).fillna(0).astype(int))
ss=pd.read_csv(R+'SampleSubmission.csv'); print('ss same order', (ss.ID.values==te.ID.values).all())
