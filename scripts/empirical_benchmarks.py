"""Fixed published correlations on exactly the R1 outer test records.

Robertson (2009) checked against the author's paper, doi:10.1139/T09-065.
Other equations transcribed from Mahler et al. (2026), Table 4:
https://doi.org/10.1007/s10706-026-03863-7
No coefficients are fitted or selected using this database. These historical
relationships use subsets of the same CPT inputs; they do not use profile
features or geological categories. Domain differences remain a limitation.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from measured_vs.data.stress import rebuild_stress
from review_round2 import scores

def predict(d):
    qt=d.qt_mpa*1000;fs=d.fs_mpa*1000
    # Robertson (2009), Eqs. 9-11, using the source study's two-pass
    # SBTn normalization. Pa = 100 kPa; all stresses are CPT-only proxies.
    net=qt-d.sigma_v_kpa
    fr=100*fs/net
    n=np.ones(len(d))
    for _ in range(2):
        qtn=(net/100)*(100/d.sigma_eff_kpa)**n
        ic=np.sqrt((3.47-np.log10(qtn))**2+(1.22+np.log10(fr))**2)
        n=np.minimum(1.,.381*ic+.05*d.sigma_eff_kpa/100-.15)
    robertson=np.sqrt(10**(.55*ic+1.68)*net/100)
    return pd.DataFrame({
        'Robertson_2009':robertson,
        'Mayne_2006':118.8*np.log10(fs)+18.5,
        'Hegazy_Mayne_1995':(10.1*np.log10(qt)-11.4)**1.67*(100*fs/qt)**.3,
        'Piratheepan_Andrus_2002':32.3*(d.qc_mpa*1000)**.089*fs**.121*d.z_mid_m**.215,
    },index=d.index)

def main():
    d=rebuild_stress(pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv'))
    d['row_id']=np.arange(len(d))
    p=pd.read_csv(ROOT/'results/revised_stress/deployable_42/outer_predictions.csv')
    values=predict(d);models=list(values)
    d=pd.concat([d[['row_id']],values],axis=1)
    p=p.merge(d,on='row_id',validate='one_to_one')
    assert np.isfinite(p[models]).all().all() and (p[models]>0).all().all()
    out=ROOT/'results/review_round2';out.mkdir(exist_ok=True)
    p.to_csv(out/'empirical_benchmark_predictions.csv',index=False)
    pd.DataFrame(scores(p,models)).to_csv(out/'empirical_benchmark_metrics.csv',index=False)
    rows=[]
    for g,f in p.groupby('group_project'):
        for r in scores(f,models):rows.append(dict(project=g,**r))
    pd.DataFrame(rows).to_csv(out/'empirical_benchmark_projects.csv',index=False)

if __name__=='__main__':main()
