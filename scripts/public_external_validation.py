"""Frozen external assessment of an explicitly age-free companion model.

The Budapest primary model requires geological age, which the public Austrian
dataset does not report per interval. This script evaluates a separate variant
with that input and its specialist/weighting removed. It does not silently
assign ages or retrain on external targets.

Source: Oberhollenzer et al., doi:10.1016/j.dib.2020.106618 (CC BY).
Raw source archive is a separate local download, not redistributed here.
"""
from pathlib import Path
import sys,json,argparse,hashlib,zipfile,platform,importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import run_nested_validation as nv
from measured_vs.data.stress import rebuild_stress
from measured_vs.data.features import engineer_profile_features
from measured_vs.utils.cache import analysis_signature, protect_cache
from review_round2 import scores

def prepare(archive):
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('.csv'))
        pieces=[]
        for c in pd.read_csv(z.open(name),chunksize=250000,low_memory=False):
            pieces.append(c[c.test_type.eq('SCPTu')])
    raw=pd.concat(pieces,ignore_index=True)
    rows=[];flags=[]
    for sounding,f in raw.groupby('ID'):
        f=f.sort_values('Depth (m)')
        # Recover the hydrostatic groundwater assumption from source u0.
        # It is not an independent field observation of groundwater.
        hydro=f[f['u0 (kPa)'].gt(0)&f['Depth (m)'].gt(0)]
        if hydro.empty:
            flags.append(dict(sounding=int(sounding),reason='No source hydrostatic pressure'));continue
        implied=hydro['Depth (m)']-hydro['u0 (kPa)']/9.81
        gw=max(0.,float(implied.median()))
        for _,v in f[f['Vs (m/s)'].notna() & ~f['Vs (m/s)'].gt(0)].iterrows():
            flags.append(dict(sounding=int(sounding),depth=float(v['Depth (m)']),reason='Nonpositive published Vs; not a usable target'))
        vs=f[f['Vs (m/s)'].gt(0)]
        previous=None
        for _,v in vs.iterrows():
            bottom=float(v['Depth (m)'])
            # Subsequent Vs measurements define consecutive receiver intervals.
            # For the first, use the published nominal 0.5 m measurement spacing.
            top=max(0.,bottom-.5) if previous is None else previous
            previous=bottom
            if bottom<=top:flags.append(dict(sounding=int(sounding),depth=bottom,reason='Nonadvancing Vs depth'));continue
            part=f[f['Depth (m)'].gt(top)&f['Depth (m)'].le(bottom)]
            cols=['qc (MPa)','qt (MPa)','fs (kPa)','u2 (kPa)']
            means=part[cols].mean()
            if not np.isfinite(means).all() or not means[['qc (MPa)','qt (MPa)','fs (kPa)']].gt(0).all():
                flags.append(dict(sounding=int(sounding),depth=bottom,reason='Missing or nonpositive interval CPT inputs'));continue
            region=str(v.basin_valley)
            rows.append(dict(project=region,group_project=region,group_cpt=f'AT_{sounding}',cpt_id=f'AT_{sounding}',z_top_m=top,z_bot_m=bottom,z_mid_m=(top+bottom)/2,gwl_m=gw,geo_age='AGE_NOT_USED',test_method='SCPT',qc_mpa=float(means['qc (MPa)']),qt_mpa=float(means['qt (MPa)']),fs_mpa=float(means['fs (kPa)'])/1000,u2_mpa=float(means['u2 (kPa)'])/1000,rf_pct=float(100*means['fs (kPa)']/1000/means['qc (MPa)']),vs_meas_mps=float(v['Vs (m/s)'])))
    data=pd.DataFrame(rows)
    data['log_vs_meas']=np.log(data.vs_meas_mps)
    return data,pd.DataFrame(flags)

def external_metrics(p):
    rows=[]
    for m in ['empirical','rf','et','base_stack','specialist_stack']:
        sound=[nv.regression_metrics_vs(f.vs_meas_mps,f[m])['rmse_mps'] for _,f in p.groupby('group_cpt')]
        rows.append(dict(model=m,mean_sounding_rmse=float(np.mean(sound)),**nv.regression_metrics_vs(p.vs_meas_mps,p[m])))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source_zip',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT/'results/public_external')
    args=ap.parse_args()
    out=args.output.resolve()
    protect_cache(out, analysis_signature(ROOT, {
        'training_table': ROOT/'data/cleaned/cpt_vs_labeled.csv', 'source_zip': args.source_zip},
        ['run_nested_validation.py', 'scripts/review_round2.py', 'scripts/public_external_validation.py'],
        {'seed':42,'trees':300}))
    ext,flags=prepare(args.source_zip)
    ext.to_csv(out/'austrian_interval_table.csv',index=False)
    flags.to_csv(out/'source_flags.csv',index=False)
    budapest=rebuild_stress(pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv'))
    budapest['geo_age']='AGE_NOT_USED'
    budapest['row_id']=np.arange(len(budapest))
    train=engineer_profile_features(budapest)
    ext=rebuild_stress(ext)
    ext['row_id']=np.arange(len(ext))
    valid=engineer_profile_features(ext)
    if not all((out/name).exists() for name in ['budapest_age_free_metrics.csv',
            'budapest_age_free_predictions.csv', 'budapest_age_free_selections.json']):
        p=train[['row_id','group_project','group_cpt','vs_meas_mps']].copy()
        selections=[]
        for fold,(a,b) in enumerate(GroupKFold(5).split(train,groups=train.group_project),1):
            print('age-free Budapest outer',fold,flush=True)
            pred,chosen,_=nv.fit_predict(train.iloc[a],train.iloc[b],42,False,300)
            for m,v in pred.items():p.loc[b,m]=v
            selections.append(dict(fold=fold,**chosen))
        p.to_csv(out/'budapest_age_free_predictions.csv',index=False)
        pd.DataFrame(scores(p,list(pred))).to_csv(out/'budapest_age_free_metrics.csv',index=False)
        (out/'budapest_age_free_selections.json').write_text(json.dumps(selections,indent=2))
    if not all((out/name).exists() for name in ['external_predictions.csv','frozen_selection.json']):
        print('Frozen fit on all 15 Budapest projects; external targets unused',flush=True)
        pred,chosen,grid=nv.fit_predict(train,valid,42,False,300)
        p=valid[['row_id','group_project','group_cpt','z_mid_m','vs_meas_mps']].copy()
        for m,v in pred.items():p[m]=v
        p.to_csv(out/'external_predictions.csv',index=False)
        (out/'frozen_selection.json').write_text(json.dumps(chosen,indent=2))
    p=pd.read_csv(out/'external_predictions.csv')
    external_metrics(p).to_csv(out/'external_metrics.csv',index=False)
    rows=[]
    for g,f in p.groupby('group_cpt'):
        for row in external_metrics(f).to_dict('records'):rows.append(dict(sounding=g,basin=f.group_project.iloc[0],rows=len(f),**row))
    pd.DataFrame(rows).to_csv(out/'external_sounding_metrics.csv',index=False)
    rows=[]
    for label,mask in [('Vs_lt200',p.vs_meas_mps.lt(200)),('Vs_200_400',p.vs_meas_mps.ge(200)&p.vs_meas_mps.lt(400)),('Vs_400_500',p.vs_meas_mps.ge(400)&p.vs_meas_mps.lt(500)),('Vs_ge500',p.vs_meas_mps.ge(500))]:
        if mask.any():
            for row in external_metrics(p[mask]).to_dict('records'):rows.append(dict(subset=label,rows=int(mask.sum()),soundings=int(p.loc[mask,'group_cpt'].nunique()),**row))
    pd.DataFrame(rows).to_csv(out/'external_velocity_metrics.csv',index=False)
    # A forward-only test: no external target enters normalization, fitting,
    # model choice, thresholds or weighting. Group names do not overlap.
    nv.assert_disjoint(train,valid)
    manifest=dict(source='https://doi.org/10.1016/j.dib.2020.106618',source_sha256=hashlib.sha256(args.source_zip.read_bytes()).hexdigest(),training_data_sha256=hashlib.sha256((ROOT/'data/cleaned/cpt_vs_labeled.csv').read_bytes()).hexdigest(),training_projects=15,external_soundings=int(valid.group_cpt.nunique()),external_basins=int(valid.group_project.nunique()),external_rows=len(valid),excluded_intervals=len(flags),seed=42,trees=300,python=platform.python_version(),packages={x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','scipy']},limitations=['Age-free companion, not original age-aware primary model','Source anonymization prevents independent project identification','Receiver interval support inferred from consecutive Vs depths; nominal 0.5 m first interval','Groundwater inferred from source hydrostatic pressure, not verified field water levels','Original field waveforms and timing picks not independently reprocessed','External results must not be used to select a preferred variant'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
