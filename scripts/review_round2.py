"""Additional diagnostics for the second review, without changing R1 results.

Run from any directory. All refits use the original seed, tree count and
inner selection; scenarios are sensitivity checks, not model selection.
"""
from pathlib import Path
import sys, json, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import run_nested_validation as nv
from measured_vs.data.features import engineer_profile_features
from measured_vs.data.stress import rebuild_stress

OUT = ROOT / 'results/review_round2'
OUT.mkdir(parents=True, exist_ok=True)

def stress_variant(raw, mode):
    d = rebuild_stress(raw)
    g = d.gamma_sat_kn_m3.copy()
    if mode == 'linear_depth':
        g += .33 * (d.z_mid_m - np.log10(d.z_mid_m))
    delta = {'unsat_0': 0., 'unsat_2': 2.}.get(mode, 1.)
    d['gamma_sat_kn_m3'] = g
    d['gamma_unsat_kn_m3'] = g-delta
    d['sigma_v_kpa'] = (g-delta)*np.minimum(d.z_mid_m, d.gwl_m.clip(lower=0)) + g*(d.z_mid_m-d.gwl_m).clip(lower=0)
    if mode == 'integrated_proxy':
        # Trapezoidal integration between midpoint unit-weight estimates.
        # Extend the shallowest estimate to ground level. This avoids treating
        # inconsistent archived interval bounds as measured layer geometry.
        for _, f in d.groupby('group_cpt'):
            total = 0.; previous = 0.; previous_g = None
            for i, r in f.sort_values('z_mid_m').iterrows():
                mid=r.z_mid_m
                mean_g=r.gamma_sat_kn_m3 if previous_g is None else .5*(previous_g+r.gamma_sat_kn_m3)
                dry=max(0.,min(mid,r.gwl_m)-previous)
                total += mean_g*(mid-previous)-dry
                d.loc[i,'sigma_v_kpa']=total
                previous=mid;previous_g=r.gamma_sat_kn_m3
    d['sigma_eff_kpa'] = d.sigma_v_kpa-d.u0_kpa
    den = d.qt_mpa*1000-d.sigma_v_kpa
    d['bq'] = (d.u2_mpa*1000-d.u0_kpa)/den.where(den > 0)
    return d

def scores(frame, models):
    rows=[]
    for m in models:
        pm = [nv.regression_metrics_vs(f.vs_meas_mps, f[m])['rmse_mps'] for _,f in frame.groupby('group_project')]
        rows.append(dict(model=m, mean_project_rmse=float(np.mean(pm)), **nv.regression_metrics_vs(frame.vs_meas_mps,frame[m])))
    return rows

def diagnostics(raw):
    p = pd.read_csv(ROOT/'results/revised_stress/deployable_42/outer_predictions.csv')
    sel = json.loads((ROOT/'results/revised_stress/deployable_42/selection.json').read_text())
    routing=[]
    for s in sel:
        f=p[p.fold.eq(s['fold'])]
        for cutoff in [400,500]:
            v=f[f.vs_meas_mps.ge(cutoff)]
            gate=v.base_stack.ge(s['threshold'])
            active=gate & (s['specialist_weights'][2]>0)
            routing.append(dict(fold=s['fold'],cutoff=cutoff,n=len(v),gate_eligible=int(gate.sum()),active_routed=int(active.sum()),threshold=s['threshold'],weight=s['specialist_weights'][2]))
    pd.DataFrame(routing).to_csv(OUT/'routing.csv',index=False)
    high=[]
    cols=['qc_mpa','fs_mpa','u2_mpa','z_mid_m','sigma_eff_kpa']
    r=rebuild_stress(raw);r['row_id']=np.arange(len(r))
    for s in sel:
        tr=r[r.group_project.isin(s['train_projects'])]
        te=r[r.group_project.isin(s['test_projects'])].merge(p[['row_id','specialist_stack']],on='row_id')
        te['outside_any_training_range']=((te[cols]<tr[cols].min())|(te[cols]>tr[cols].max())).any(axis=1)
        for (project,age),f in te[te.vs_meas_mps.ge(400)].groupby(['group_project','geo_age']):
            for cutoff in [400,500]:
                v=f[f.vs_meas_mps.ge(cutoff)]
                if len(v):high.append(dict(project=project,geo_age=age,cutoff=cutoff,n=len(v),outside_any_training_range=int(v.outside_any_training_range.sum()),**nv.regression_metrics_vs(v.vs_meas_mps,v.specialist_stack)))
    pd.DataFrame(high).to_csv(OUT/'high_velocity_projects.csv',index=False)
    coverage=[]
    for k in [12,15]:
        lo=0 if k==0 else beta.ppf(.025,k,16-k)
        hi=1 if k==15 else beta.ppf(.975,k+1,15-k)
        coverage.append(dict(covered=k,projects=15,lower=lo,upper=hi,interpretation='Clopper-Pearson reference under independent Bernoulli projects; overlap makes this descriptive, not a calibrated regional confidence interval'))
    pd.DataFrame(coverage).to_csv(OUT/'coverage_reference_intervals.csv',index=False)
    thickness=raw.z_bot_m-raw.z_top_m
    raw.loc[thickness.le(0),['group_project','group_cpt','z_top_m','z_bot_m','z_mid_m']].to_csv(OUT/'interval_bound_flags.csv',index=False)
    summary={'nonpositive_thickness_records':int(thickness.le(0).sum()),'positive_thickness_m':thickness[thickness.gt(0)].describe(percentiles=[.05,.25,.5,.75,.95]).to_dict()}
    spans={}
    for w in [3,5]:
        vals=[]
        for _,f in raw.groupby('group_cpt'):
            f=f.sort_values('z_mid_m').reset_index(drop=True)
            for i in range(len(f)):
                a=max(0,i-w//2);b=min(len(f),i+w//2+1)
                vals.append(f.z_bot_m.iloc[b-1]-f.z_top_m.iloc[a])
        spans[str(w)]=pd.Series(vals).describe(percentiles=[.05,.5,.95]).to_dict()
    summary['window_span_m']=spans
    (OUT/'diagnostics.json').write_text(json.dumps(summary,indent=2))
    # Existing component predictions are paired ablations, with no refitting.
    pd.DataFrame(scores(p,['empirical','rf','et','base_stack','specialist_stack'])).to_csv(OUT/'component_ablation.csv',index=False)

def refit(raw, name, lopo=False):
    target=OUT/name
    if (target/'metrics.csv').exists():return
    target.mkdir(exist_ok=True)
    d=stress_variant(raw,name)
    windows={'window_1':(1,), 'window_5_7':(5,7)}.get(name,(3,5))
    d=engineer_profile_features(d,profile_windows=windows)
    if name=='no_profile':
        d=d.drop(columns=[c for c in d if any(t in c for t in ['_roll','_diff1','_local_resid']) or c in ['depth_rank_pct','z_rel_minmax']])
    d['row_id']=np.arange(len(d))
    oof=d[['row_id','group_project','group_cpt','geo_age','vs_meas_mps']].copy()
    selections=[]
    for fold,(a,b) in enumerate(GroupKFold(15 if lopo else 5).split(d,groups=d.group_project),1):
        print(name,fold,flush=True)
        pred,s,_=nv.fit_predict(d.iloc[a],d.iloc[b],42,False,300)
        for m,v in pred.items():oof.loc[b,m]=v
        oof.loc[b,'fold']=fold
        selections.append(dict(fold=fold,**s))
        oof.to_csv(target/'predictions.csv',index=False)
        (target/'selection.json').write_text(json.dumps(selections,indent=2))
    pd.DataFrame(scores(oof,list(pred))).to_csv(target/'metrics.csv',index=False)

def main():
    raw=pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv')
    diagnostics(raw)
    for name in ['linear_depth','unsat_0','unsat_2','integrated_proxy','window_1','window_5_7','no_profile','leave_one_project_out']:
        refit(raw,name,lopo=name=='leave_one_project_out')
    manifest=dict(data_sha256=hashlib.sha256((ROOT/'data/cleaned/cpt_vs_labeled.csv').read_bytes()).hexdigest(),seed=42,trees=300,python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
