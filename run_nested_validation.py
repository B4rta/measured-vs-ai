"""Evaluate all blend and gate selection inside project-grouped outer folds.

The original training script is retained for historical reproduction only.
This entry point produces predictions and audit tables, not fitted-data scores.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from measured_vs.data.features import engineer_profile_features, tree_feature_columns, compute_sample_weights
from measured_vs.data.stress import rebuild_stress
from measured_vs.models.baseline import EmpiricalBaselineModel
from measured_vs.models.trees import TreeProfileModel
from measured_vs.evaluation.metrics import regression_metrics_vs

THRESHOLDS = (325., 350., 375.)
CAPS = (.50, .65, .80)


def assert_disjoint(*frames):
    groups = [set(f.group_project) for f in frames]
    for i, a in enumerate(groups):
        for b in groups[i+1:]:
            if a & b:
                raise ValueError("A project occurs in more than one partition")


def components(train, valid, seed, modality, trees):
    assert_disjoint(train, valid)
    cols, cats = tree_feature_columns(train, modality)
    cols = [c for c in cols if c != "row_id"]
    if not modality:
        assert not ({"test_method", "age_method"} & set(cols))
    y = np.log(train.vs_meas_mps.to_numpy())
    emp = EmpiricalBaselineModel().fit(train, y)
    pred = pd.DataFrame(index=valid.index)
    pred["empirical"] = emp.predict_log_vs(valid)
    def tree(part, kind, offset, leaf):
        m = TreeProfileModel(kind, cols, cats, trees, "sqrt", leaf, seed+offset)
        w = compute_sample_weights(part, scpt_boost=.3 if modality else 0.)
        return m.fit(part, np.log(part.vs_meas_mps.to_numpy()), w).predict_log_vs(valid)
    pred["rf"] = tree(train, "rf", 0, 2)
    pred["et"] = tree(train, "et", 1, 2)
    specs = [("tertiary", train.geo_age.eq("Tertiary"), "rf", 10, 2, 60)]
    if modality:
        specs.append(("scpt", train.test_method.eq("SCPT"), "rf", 20, 1, 40))
    specs += [(f"high_{int(t)}", train.vs_meas_mps.ge(t), "et", 30, 1, 80) for t in THRESHOLDS]
    for name, mask, kind, offset, leaf, minimum in specs:
        pred[name] = tree(train.loc[mask], kind, offset, leaf) if mask.sum() >= minimum else np.nan
    return pred


def base_blend(p, w):
    return p[["empirical", "rf", "et"]].to_numpy() @ np.asarray(w)


def specialist_blend(p, frame, base, threshold, weights):
    out = base.copy()
    gates = [frame.geo_age.eq("Tertiary").to_numpy(), frame.test_method.eq("SCPT").to_numpy(), np.exp(base) >= threshold]
    for name, gate, weight in zip(["tertiary", "scpt", f"high_{int(threshold)}"], gates, weights):
        if name in p and weight:
            v = p[name].to_numpy()
            mask = gate & np.isfinite(v)
            out[mask] = (1-weight)*out[mask] + weight*v[mask]
    return out


def select(p, frame, modality):
    y = frame.vs_meas_mps.to_numpy()
    best = (np.inf, None)
    for i in range(21):
        for j in range(21-i):
            w = (i/20, j/20, (20-i-j)/20)
            mse = np.mean((np.exp(base_blend(p, w))-y)**2)
            if mse < best[0]:
                best = (mse, w)
    base = base_blend(p, best[1])
    choices = []
    for threshold in THRESHOLDS:
        for cap in CAPS:
            winner = (np.inf, None)
            grid = np.arange(round(cap*20)+1)/20
            for wt in grid:
                for ws in (grid if modality else [0.]):
                    # Vectorise the last blend dimension for the search.
                    first = specialist_blend(p, frame, base, threshold, (wt, ws, 0.))
                    v = p[f"high_{int(threshold)}"].to_numpy()
                    mask = (np.exp(base) >= threshold) & np.isfinite(v)
                    candidates = np.repeat(first[:, None], len(grid), axis=1)
                    candidates[mask] = first[mask, None]*(1-grid) + v[mask, None]*grid
                    errors = np.mean((np.exp(candidates)-y[:, None])**2, axis=0)
                    k = int(np.argmin(errors))
                    score = float(errors[k])
                    weights = (float(wt), float(ws), float(grid[k]))
                    if score < winner[0]-1e-12 or (abs(score-winner[0]) <= 1e-12 and sum(weights) < sum(winner[1])):
                        winner = (score, weights)
            choices.append(dict(threshold=threshold, cap=cap, inner_rmse=float(np.sqrt(winner[0])), specialist_weights=winner[1]))
    # A smaller cap and smaller correction wins an exact numerical tie.
    chosen = min(choices, key=lambda c: (round(c["inner_rmse"], 12), c["cap"], sum(c["specialist_weights"]), c["threshold"]))
    return dict(base_weights=best[1], **chosen), choices


def fit_predict(train, valid, seed, modality, trees, inner_folds=3):
    assert_disjoint(train, valid)
    train = train.reset_index(drop=True)
    p = None
    splits = GroupKFold(min(inner_folds, train.group_project.nunique()))
    for a, b in splits.split(train, groups=train.group_project):
        pp = components(train.iloc[a], train.iloc[b], seed, modality, trees)
        if p is None:
            p = pd.DataFrame(np.nan, index=train.index, columns=pp.columns)
        p.loc[b, pp.columns] = pp.to_numpy()
    chosen, grid = select(p, train, modality)
    outer = components(train, valid, seed, modality, trees)
    base = base_blend(outer, chosen["base_weights"])
    result = {name: np.exp(outer[name].to_numpy()) for name in ["empirical", "rf", "et"]}
    result["base_stack"] = np.exp(base)
    result["specialist_stack"] = np.exp(specialist_blend(outer, valid, base, chosen["threshold"], chosen["specialist_weights"]))
    return result, chosen, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--output", default="results/revised_stress")
    ap.add_argument("--modalities", nargs="+", choices=["deployable", "retrospective"], default=["deployable", "retrospective"])
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = root/args.output
    out.mkdir(parents=True, exist_ok=True)
    source = root/"data/cleaned/cpt_vs_labeled.csv"
    raw = rebuild_stress(pd.read_csv(source))
    if not np.isfinite(raw.vs_meas_mps).all() or not raw.vs_meas_mps.gt(0).all():
        raise ValueError("Targets must be finite and positive")
    raw["row_id"] = np.arange(len(raw))
    audit = []
    for c in raw:
        audit.append(dict(column=c, missing=int(raw[c].isna().sum()), missing_pct=100*raw[c].isna().mean(), nonpositive=int((raw[c]<=0).sum()) if pd.api.types.is_numeric_dtype(raw[c]) else None))
    pd.DataFrame(audit).to_csv(out/"data_audit.csv", index=False)
    metadata = dict(arguments=vars(args), data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), python=platform.python_version(), packages={p:importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy"]}, rows=len(raw), projects=raw.group_project.nunique(), soundings=raw.group_cpt.nunique(), duplicate_rows=int(raw.drop(columns="row_id").duplicated().sum()), removed_rows=0)
    (out/"run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for mode in args.modalities:
        modality = mode == "retrospective"
        # Keep original modality for subgroup reports; exclude it explicitly from X.
        data = engineer_profile_features(raw, include_test_method_for_trees=True)
        for seed in args.seeds:
            path = out/f"{mode}_{seed}"
            path.mkdir(exist_ok=True)
            if (path/"complete.json").exists():
                continue
            oof = data[["row_id", "project", "group_project", "group_cpt", "geo_age", "test_method", "z_mid_m", "vs_meas_mps"]].copy()
            selections, grids, folds = [], [], []
            for fold, (tr, te) in enumerate(GroupKFold(5).split(data, groups=data.group_project), 1):
                train, test = data.iloc[tr], data.iloc[te]
                print(f"{mode} seed={seed} outer={fold}: {train.group_project.nunique()} train projects, {test.group_project.nunique()} test projects", flush=True)
                pred, chosen, grid = fit_predict(train, test, seed, modality, args.trees)
                selections.append(dict(fold=fold, train_projects=sorted(train.group_project.unique()), test_projects=sorted(test.group_project.unique()), **chosen))
                grids += [dict(fold=fold, **r) for r in grid]
                oof.loc[te, "fold"] = fold
                for name, values in pred.items():
                    oof.loc[te, name] = values
                    folds.append(dict(fold=fold, model=name, n=len(te), **regression_metrics_vs(test.vs_meas_mps, values)))
                oof.to_csv(path/"outer_predictions.csv", index=False)
                (path/"selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
            models = ["empirical", "rf", "et", "base_stack", "specialist_stack"]
            pd.DataFrame([dict(model=m, **regression_metrics_vs(oof.vs_meas_mps, oof[m])) for m in models]).to_csv(path/"pooled_metrics.csv", index=False)
            pm = pd.DataFrame([dict(project=g, model=m, n=len(f), **regression_metrics_vs(f.vs_meas_mps, f[m])) for g,f in oof.groupby("group_project") for m in models])
            pm.to_csv(path/"project_metrics.csv", index=False)
            pd.DataFrame(folds).to_csv(path/"fold_metrics.csv", index=False)
            pd.DataFrame(grids).to_csv(path/"inner_sensitivity.csv", index=False)
            (path/"complete.json").write_text(json.dumps(dict(seed=seed, mode=mode, trees=args.trees)))
    print("Nested validation complete", flush=True)


if __name__ == "__main__":
    main()
