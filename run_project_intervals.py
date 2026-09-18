"""Random-project split conformal evaluation, separate from point CV.

Each of five splits has 3 test, 9 calibration and 3 proper-training projects.
Project maximum residuals target simultaneous coverage of a new project.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from run_nested_validation import fit_predict, assert_disjoint
from measured_vs.data.features import engineer_profile_features
from measured_vs.data.stress import rebuild_stress
from measured_vs.evaluation.project_conformal import project_quantile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trees", type=int, default=300)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    out = root/"results/revised_intervals"
    out.mkdir(parents=True, exist_ok=True)
    data = rebuild_stress(pd.read_csv(root/"data/cleaned/cpt_vs_labeled.csv"))
    data["row_id"] = np.arange(len(data))
    data = engineer_profile_features(data)
    groups = np.array(sorted(data.group_project.unique()))
    np.random.default_rng(20260918).shuffle(groups)
    blocks = np.array_split(groups, 5)
    results, audit, coverage = [], [], []
    for fold, test_groups in enumerate(blocks, 1):
        others = groups[~np.isin(groups, test_groups)].copy()
        np.random.default_rng(1700+fold).shuffle(others)
        train = data[data.group_project.isin(others[:3])]
        cal = data[data.group_project.isin(others[3:])]
        test = data[data.group_project.isin(test_groups)]
        assert_disjoint(train, cal, test)
        print(f"Conformal fold {fold}: training {len(train)}, calibration {len(cal)}, test {len(test)} rows", flush=True)
        pred, chosen, _ = fit_predict(train, pd.concat([cal,test]), 42, False, args.trees)
        pred = pred["specialist_stack"]
        cp, tp = pred[:len(cal)], pred[len(cal):]
        scores = pd.DataFrame({"project":cal.group_project.to_numpy(), "score":abs(cal.vs_meas_mps.to_numpy()-cp)}).groupby("project").score.max()
        r = test[["row_id","group_project","test_method","geo_age","vs_meas_mps"]].copy()
        r["prediction"] = tp
        r["fold"] = fold
        for alpha in [.1,.2]:
            pct = round((1-alpha)*100)
            q = project_quantile(scores.to_numpy(), alpha)
            r[f"lower_{pct}"] = np.maximum(0,tp-q)
            r[f"upper_{pct}"] = tp+q
            r[f"covered_{pct}"] = abs(r.vs_meas_mps-tp) <= q
            for g,f in r.groupby("group_project"):
                coverage.append(dict(fold=fold, project=g, nominal=pct, half_width=q, n=len(f), row_coverage=f[f"covered_{pct}"].mean(), whole_project_covered=bool(f[f"covered_{pct}"].all()), mean_width=float((f[f"upper_{pct}"]-f[f"lower_{pct}"]).mean())))
        results.append(r)
        audit.append(dict(fold=fold, training_projects=sorted(train.group_project.unique()), calibration_projects=sorted(cal.group_project.unique()), test_projects=sorted(test.group_project.unique()), calibration_scores=scores.to_dict(), selection=chosen))
        pd.concat(results).to_csv(out/"predictions.csv",index=False)
        pd.DataFrame(coverage).to_csv(out/"coverage.csv",index=False)
        (out/"splits.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
