"""Build project diagnostics and manuscript figures from outer predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/revision_summary"
FIG=ROOT/"docs/figures"
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":220})
df=pd.read_csv(ROOT/"results/revised_stress/deployable_42/outer_predictions.csv")
pm=pd.read_csv(ROOT/"results/revised_stress/deployable_42/project_metrics.csv")
models=["empirical","rf","et","base_stack","specialist_stack"]
names={"empirical":"Robust baseline","rf":"Random Forest","et":"Extra Trees","base_stack":"Base stack","specialist_stack":"Specialist stack"}
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/f"{name}.png",bbox_inches="tight");plt.close(fig)
def rmse(a,b):return np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2))

macro=pm.groupby("model").agg(mean_project_rmse=("rmse_mps","mean"),median_project_rmse=("rmse_mps","median"),min_project_rmse=("rmse_mps","min"),max_project_rmse=("rmse_mps","max"))
macro.to_csv(OUT/"macro_metrics.csv")
pivot=pm.pivot(index="project",columns="model",values="rmse_mps")
delta=pivot.specialist_stack-pivot.empirical
rng=np.random.default_rng(1701)
boot=np.mean(rng.choice(delta.to_numpy(),size=(10000,len(delta)),replace=True),axis=1)
influence=[]
for g in df.group_project.unique():
    s=df[df.group_project.ne(g)]
    influence.append(dict(omitted_project=g,baseline_rmse=rmse(s.vs_meas_mps,s.empirical),stack_rmse=rmse(s.vs_meas_mps,s.specialist_stack),delta_rmse=rmse(s.vs_meas_mps,s.specialist_stack)-rmse(s.vs_meas_mps,s.empirical)))
pd.DataFrame(influence).to_csv(OUT/"project_influence.csv",index=False)
summary={"paired_mean_delta_rmse":float(delta.mean()),"paired_median_delta_rmse":float(delta.median()),"projects_improved":int((delta<0).sum()),"paired_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),"influence_delta_range":[min(x["delta_rmse"] for x in influence),max(x["delta_rmse"] for x in influence)]}
seed_rows=[]
for p in (ROOT/"results/revised_stress").glob("*/pooled_metrics.csv"):
    f=pd.read_csv(p);f["setting"]=p.parent.name;seed_rows.append(f)
pd.concat(seed_rows).to_csv(OUT/"seed_metrics.csv",index=False)

fig,axs=plt.subplots(1,2,figsize=(9,4))
for ax,m in zip(axs,["empirical","specialist_stack"]):
    ax.scatter(df.vs_meas_mps,df[m],s=9,alpha=.30,color="#245677")
    ax.plot([100,750],[100,750],"--",color="black",lw=1)
    ax.set(xlabel="Measured Vs (m/s)",ylabel="Predicted Vs (m/s)",title=names[m],xlim=(100,750),ylim=(100,750))
save(fig,"fig03_outer_scatter")

fig,ax=plt.subplots(figsize=(8,5.4));y=np.arange(len(pivot))
ax.barh(y-.17,pivot.empirical,height=.32,label="Robust baseline",color="#9aa5ac")
ax.barh(y+.17,pivot.specialist_stack,height=.32,label="Specialist stack",color="#245677")
ax.set(yticks=y,yticklabels=pivot.index,xlabel="Held-out-project RMSE (m/s)");ax.legend();save(fig,"fig04_project_errors")

bins=[0,200,250,300,350,400,500,1000]
df["bin"]=pd.cut(df.vs_meas_mps,bins,right=False)
rows=[]
for label,f in df.groupby("bin",observed=True):
    for m in ["empirical","specialist_stack"]:
        # Bootstrap projects, preserving all rows of each selected project.
        groups=list(f.group_project.unique())
        sums=np.array([(f.loc[f.group_project.eq(g),m]-f.loc[f.group_project.eq(g),"vs_meas_mps"]).sum() for g in groups])
        counts=np.array([f.group_project.eq(g).sum() for g in groups])
        draws=rng.integers(0,len(groups),size=(2000,len(groups)))
        estimates=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        rows.append(dict(bin=str(label),model=m,n=len(f),projects=len(groups),mean_measured=f.vs_meas_mps.mean(),bias=(f[m]-f.vs_meas_mps).mean(),rmse=rmse(f.vs_meas_mps,f[m]),low=np.quantile(estimates,.025),high=np.quantile(estimates,.975)))
b=pd.DataFrame(rows);b.to_csv(OUT/"velocity_bins.csv",index=False)
fig,ax=plt.subplots(figsize=(8,4.4))
for m,col in [("empirical","#7b858d"),("specialist_stack","#245677")]:
    f=b[b.model.eq(m)]
    ax.plot(f.mean_measured,f.bias,"o-",label=names[m],color=col)
    ax.fill_between(f.mean_measured,f.low,f.high,alpha=.14,color=col)
for _,r in b[b.model.eq("specialist_stack")].iterrows():ax.annotate(f"n={r.n}",(r.mean_measured,r.bias),xytext=(0,12),textcoords="offset points",ha="center",fontsize=8)
ax.axhline(0,color="black",lw=.7);ax.set(xlabel="Mean measured Vs in bin (m/s)",ylabel="Prediction minus measured Vs (m/s)");ax.legend();save(fig,"fig05_residual_bins")
cal=[]
for m in models:
    slope,intercept=np.polyfit(df[m],df.vs_meas_mps,1)
    cal.append(dict(model=m,calibration_slope=slope,calibration_intercept_mps=intercept))
pd.DataFrame(cal).to_csv(OUT/"calibration.csv",index=False)
summary["calibration"]=cal

cov=pd.read_csv(ROOT/"results/revised_intervals/coverage.csv")
cp=pd.read_csv(ROOT/"results/revised_intervals/predictions.csv")
cs=[]
for pct in [80,90]:
    c=cov[cov.nominal.eq(pct)]
    cs.append(dict(nominal=pct,pooled_coverage=cp[f"covered_{pct}"].mean(),macro_coverage=c.row_coverage.mean(),whole_project_coverage=c.whole_project_covered.mean(),mean_width=(cp[f"upper_{pct}"]-cp[f"lower_{pct}"]).mean(),min_halfwidth=c.half_width.min(),max_halfwidth=c.half_width.max()))
pd.DataFrame(cs).to_csv(OUT/"conformal_summary.csv",index=False)
fig,ax=plt.subplots(figsize=(8,4.3))
c=cov[cov.nominal.eq(90)].sort_values("project")
ax.bar(np.arange(len(c)),100*c.row_coverage,color="#245677");ax.axhline(90,color="black",ls="--",lw=1)
ax.set(xticks=np.arange(len(c)),xticklabels=c.project,ylim=(0,105),ylabel="Held-out interval coverage (%)")
plt.setp(ax.get_xticklabels(),rotation=65,ha="right");save(fig,"fig06_project_coverage")

raw=pd.read_csv(ROOT/"results/source_audit/revised_modelling_table.csv")
fig,axs=plt.subplots(1,4,figsize=(9,5),sharey=True)
for i,method in enumerate(["SCPT","MASW"]):
    sub=raw[raw.test_method.eq(method)]
    # Most populated profile per category, selected without residuals.
    g=sub.groupby("group_cpt").size().idxmax();p=sub[sub.group_cpt.eq(g)].sort_values("z_mid_m")
    for j,(v,label) in enumerate([("qc_mpa","qc (MPa)"),("fs_mpa","fs (MPa)"),("u2_mpa","u2 (MPa)"),("vs_meas_mps","Measured Vs (m/s)")]):
        axs[j].plot(p[v],p.z_mid_m,".-",label=g,color=["#245677","#b26839"][i]);axs[j].set_xlabel(label)
axs[0].invert_yaxis();axs[0].set_ylabel("Depth below ground (m)");axs[3].legend(fontsize=7,loc="lower right");save(fig,"fig02_profiles")

subsets={"All":np.ones(len(df),bool),"Quaternary":df.geo_age.eq("Quaternary"),"Tertiary":df.geo_age.eq("Tertiary"),"Surface-wave":df.test_method.eq("MASW"),"SCPT":df.test_method.eq("SCPT"),"Vs >= 400":df.vs_meas_mps.ge(400),"Vs < 400":df.vs_meas_mps.lt(400)}
sub=[]
for name,mask in subsets.items():
    f=df[mask]
    for m in models:sub.append(dict(subset=name,model=m,n=len(f),rmse=rmse(f.vs_meas_mps,f[m]),bias=(f[m]-f.vs_meas_mps).mean()))
pd.DataFrame(sub).to_csv(OUT/"subset_metrics.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
