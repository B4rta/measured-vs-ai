"""Fixed-test training-project influence and controlled profile perturbations.

These are additional sensitivity analyses on existing Budapest projects.
They do not create independent sites or recover raw CPT measurements.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import run_nested_validation as nv
from measured_vs.data.stress import rebuild_stress
from measured_vs.data.features import engineer_profile_features
from review_round2 import scores

OUT=ROOT/'results/followup_validation'

def physical_windows(raw):
    d=engineer_profile_features(raw)
    for _,ix in d.groupby('group_cpt').groups.items():
        f=d.loc[ix];z=f.z_mid_m.to_numpy()
        for col in ['log_qc_mpa','log_qt_mpa','log_fs_mpa','log_rf_pct','log_sigma_eff_kpa']:
            x=f[col].to_numpy()
            for w,width in [(3,3.),(5,5.)]:
                means=np.array([np.nanmean(x[np.abs(z-zz)<=width/2]) for zz in z])
                d.loc[ix,f'{col}_roll{w}']=means
            d.loc[ix,f'{col}_local_resid_w3']=x-d.loc[ix,f'{col}_roll3'].to_numpy()
    return d

def resampled_context(raw, spacing):
    """Interpolate observed CPT intervals, then rebuild context on a fixed grid.

    Only context columns are mapped back to the original scoring rows. This
    cannot recover sub-interval structure lost during original aggregation.
    No measured Vs is interpolated or supplied to feature construction.
    """
    original=engineer_profile_features(raw)
    context=[c for c in original if any(v in c for v in ['_roll','_diff1','_local_resid']) or c in ['depth_rank_pct','z_rel_minmax']]
    for _,ix in original.groupby('group_cpt').groups.items():
        f=original.loc[ix].sort_values('z_mid_m');z=f.z_mid_m.to_numpy()
        # Coalesce duplicate depths for CPT interpolation only. Scoring rows
        # and their targets remain separate and unchanged.
        aggregations={c:('mean' if pd.api.types.is_numeric_dtype(f[c]) else 'first') for c in raw.columns if c not in ['z_mid_m','vs_meas_mps','log_vs_meas']}
        source=f.groupby('z_mid_m',as_index=False).agg(aggregations)
        source_z=source.z_mid_m.to_numpy()
        grid=np.unique(np.r_[np.arange(z[0],z[-1],spacing),z[-1]])
        g=pd.DataFrame(index=np.arange(len(grid)))
        # Start from raw data columns only; existing derived log columns would
        # otherwise be inadvertently interpolated and reused as inputs.
        for c in raw.columns:
            if c in ['vs_meas_mps','log_vs_meas']:continue
            if c=='z_mid_m':g[c]=grid
            elif pd.api.types.is_numeric_dtype(raw[c]):g[c]=np.interp(grid,source_z,source[c])
            else:g[c]=source[c].iloc[0]
        g['z_mid_m']=grid
        # Exclude targets: the feature builder never uses them, but zeroing
        # makes the no-target-interpolation contract explicit.
        g['vs_meas_mps']=1.;g['log_vs_meas']=0.
        h=engineer_profile_features(g)
        for c in context:
            valid=h[c].notna()
            original.loc[f.index,c]=np.interp(z,h.loc[valid,'z_mid_m'],h.loc[valid,c]) if valid.any() else np.nan
    return original

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=rebuild_stress(pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv'))
    raw['row_id']=np.arange(len(raw))
    d=engineer_profile_features(raw)
    reference=pd.read_csv(ROOT/'results/revised_stress/deployable_42/outer_predictions.csv')
    influence=[]
    for fold,(a,b) in enumerate(GroupKFold(5).split(d,groups=d.group_project),1):
        train,test=d.iloc[a],d.iloc[b]
        for deleted in sorted(train.group_project.unique()):
            key=f'fold{fold}_{list(sorted(train.group_project.unique())).index(deleted):02d}'
            target=OUT/'influence'/key;target.mkdir(parents=True,exist_ok=True)
            if not (target/'selection.json').exists():
                print('influence',fold,deleted,flush=True)
                fit=train[train.group_project.ne(deleted)]
                pred,chosen,_=nv.fit_predict(fit,test,42,False,300)
                p=test[['row_id','group_project','vs_meas_mps']].copy()
                for m,v in pred.items():p[m]=v
                p.to_csv(target/'predictions.csv',index=False)
                (target/'selection.json').write_text(json.dumps(dict(fold=fold,deleted_training_project=deleted,test_projects=sorted(test.group_project.unique()),**chosen),indent=2))
            p=pd.read_csv(target/'predictions.csv')
            ref=reference[reference.fold.eq(fold)]
            for row in scores(p,['empirical','base_stack','specialist_stack']):
                m=row['model'];orig=nv.regression_metrics_vs(ref.vs_meas_mps,ref[m])
                row.update(fold=fold,deleted_training_project=deleted,reference_pooled_rmse=orig['rmse_mps'],rmse_change=row['rmse_mps']-orig['rmse_mps'])
                influence.append(row)
            pd.DataFrame(influence).to_csv(OUT/'training_project_influence.csv',index=False)
    combined=[]
    for deleted in sorted(d.group_project.unique()):
        counterfactual=reference.copy().set_index('row_id')
        for job in sorted((OUT/'influence').glob('*/selection.json')):
            info=json.loads(job.read_text())
            if info['deleted_training_project']!=deleted:continue
            p=pd.read_csv(job.parent/'predictions.csv').set_index('row_id')
            for m in ['empirical','base_stack','specialist_stack']:
                counterfactual.loc[p.index,m]=p[m]
        # For the fold where deleted is a test project, the reference model
        # already excluded it from training and needs no extra refit.
        for row in scores(counterfactual.reset_index(),['empirical','base_stack','specialist_stack']):
            row['deleted_training_project']=deleted
            combined.append(row)
    pd.DataFrame(combined).to_csv(OUT/'combined_project_deletion.csv',index=False)
    for scenario in ['physical_3m_5m','missing_every_fifth','missing_every_third','resampled_context_0p5m','resampled_context_2m']:
        target=OUT/scenario;target.mkdir(exist_ok=True)
        if (target/'metrics.csv').exists():continue
        if scenario=='physical_3m_5m':data=physical_windows(raw)
        elif scenario.startswith('resampled_context_'):
            data=resampled_context(raw,.5 if scenario.endswith('0p5m') else 2.)
        else:
            # Fixed positional omissions, independent of Vs or model residuals.
            r=raw.sort_values(['group_cpt','z_mid_m']).copy()
            n=5 if scenario=='missing_every_fifth' else 3
            keep=r.groupby('group_cpt').cumcount().mod(n).ne(n-1)
            data=engineer_profile_features(r.loc[keep])
        # The original outer project allocation is retained in every scenario.
        p=data[['row_id','group_project','vs_meas_mps']].copy()
        selections=[]
        for fold in range(1,6):
            held=set(reference.loc[reference.fold.eq(fold),'group_project'])
            te=data.group_project.isin(held);tr=~te
            print(scenario,fold,flush=True)
            pred,chosen,_=nv.fit_predict(data.loc[tr],data.loc[te],42,False,300)
            for m,v in pred.items():p.loc[te,m]=v
            selections.append(dict(fold=fold,test_projects=sorted(held),**chosen))
        p.to_csv(target/'predictions.csv',index=False)
        (target/'selection.json').write_text(json.dumps(selections,indent=2))
        rows=scores(p,['empirical','base_stack','specialist_stack'])
        # Paired complete-profile reference on exactly the retained test rows.
        ref=reference[reference.row_id.isin(p.row_id)]
        for row in rows:
            m=row['model'];paired=scores(ref,[m])[0]
            row['reference_same_rows_project_rmse']=paired['mean_project_rmse']
            row['reference_same_rows_pooled_rmse']=paired['rmse_mps']
            row['rows']=len(p)
        pd.DataFrame(rows).to_csv(target/'metrics.csv',index=False)

if __name__=='__main__':main()
