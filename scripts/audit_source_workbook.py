"""Compare the archived modelling table with an optional source workbook.

Only audit summaries are written; the private workbook is not republished.
"""
from pathlib import Path
import sys
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import openpyxl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from measured_vs.data.stress import rebuild_stress

p=argparse.ArgumentParser()
p.add_argument("workbook")
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
source=Path(args.workbook)
sheet=openpyxl.load_workbook(source,read_only=True,data_only=True).worksheets[0]
records=[r for r in sheet.iter_rows(min_row=5,max_col=30,values_only=True) if isinstance(r[0],str) and isinstance(r[2],(int,float))]
old=pd.read_csv(root/"data/cleaned/cpt_vs_labeled.csv")
new=rebuild_stress(old)
assert len(records)==len(old)
mapping={"project":0,"cpt_id":1,"z_top_m":2,"z_bot_m":3,"z_mid_m":4,"gwl_m":5,"geo_age":6,"qc_mpa":7,"fs_mpa":8,"u2_mpa":9,"rf_pct":10,"vs_meas_mps":11,"test_method":13,"qt_mpa":29}
checks={}
for c,i in mapping.items():
    a=np.array([r[i] for r in records])
    checks[c]=bool(np.allclose(a.astype(float),old[c],rtol=1e-10,atol=1e-10)) if pd.api.types.is_numeric_dtype(old[c]) else bool((a==old[c]).all())
if not all(checks.values()):
    raise ValueError(checks)
gamma=np.array([r[18] for r in records],dtype=float)
assert np.allclose(gamma,new.gamma_sat_kn_m3)
out=root/"results/source_audit"
out.mkdir(parents=True,exist_ok=True)
report=dict(source_file=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),worksheet=sheet.title,worksheet_extent_rows=sheet.max_row,actual_interval_records=len(records),retained_records=len(old),removed_records=0,source_to_cleaned_checks=checks,source_cpt_only_gamma_matches_reconstruction=True,original_field_reading_count="not provided",outlier_removal="none in this revision",source_matching="15 m horizontal criterion and seismic-layer averaging documented in Mahler et al. 2026",source_formula="11.46 + 0.33*log10(z_m) + 3.1*log10(fs_kPa) + 0.7*log10(qt_kPa)",stress_assumption="homogeneous column using local CPT-only unit weight; not measured stress or cumulative layer integration")
(out/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
cols=["gamma_sat_kn_m3","gamma_unsat_kn_m3","sigma_v_kpa","sigma_eff_kpa","bq"]
pd.DataFrame([dict(variable=c,old_min=old[c].min(),old_max=old[c].max(),revised_min=new[c].min(),revised_max=new[c].max(),changed_rows=int((~np.isclose(old[c],new[c])).sum())) for c in cols]).to_csv(out/"stress_changes.csv",index=False)
new.to_csv(out/"revised_modelling_table.csv",index=False)
print(json.dumps(report,indent=2))
