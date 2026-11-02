"""Summarize the follow-up without refitting or filtering external outcomes."""
from pathlib import Path
import sys, json, hashlib, platform, importlib.metadata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from measured_vs.data.stress import rebuild_stress


def main():
    out = ROOT / 'results/followup_validation'
    rows = []
    for path in sorted(out.glob('*/metrics.csv')):
        for row in pd.read_csv(path).to_dict('records'):
            rows.append(dict(scenario=path.parent.name, **row))
    pd.DataFrame(rows).to_csv(out / 'profile_scenario_summary.csv', index=False)
    source = ROOT / 'data/cleaned/cpt_vs_labeled.csv'
    manifest = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seed=42, trees=300, outer_project_folds=5, inner_project_folds=3,
        fixed_test_training_deletion_refits=60,
        python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy']},
        limitations=[
            'All local sensitivity analyses reuse the original 15 projects',
            'Periodic omissions are simulations scored only at retained depths',
            'Context resampling interpolates archived intervals, not original raw CPT logs',
            'No variant replaces the reference model based on these results',
        ],
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    public = ROOT / 'results/public_external'
    train = rebuild_stress(pd.read_csv(source))
    ext = rebuild_stress(pd.read_csv(public / 'austrian_interval_table.csv'))
    ext['row_id'] = range(len(ext))
    columns = ['qc_mpa', 'qt_mpa', 'fs_mpa', 'u2_mpa', 'rf_pct', 'bq', 'sigma_eff_kpa', 'z_mid_m', 'gwl_m']
    ranges = []
    flags = ext[['row_id', 'group_cpt', 'z_mid_m']].copy()
    for col in columns:
        lo, hi = train[col].min(), train[col].max()
        outside = ext[col].lt(lo) | ext[col].gt(hi)
        flags[col + '_outside'] = outside
        flags[col + '_missing'] = ext[col].isna()
        ranges.append(dict(variable=col, training_min=lo, training_max=hi,
                           external_min=ext[col].min(), external_max=ext[col].max(),
                           outside_rows=int(outside.sum()), missing_rows=int(ext[col].isna().sum())))
    flags['outside_any_checked_range'] = flags[[c + '_outside' for c in columns]].any(axis=1)
    flags.to_csv(public / 'input_support_flags.csv', index=False)
    pd.DataFrame(ranges).to_csv(public / 'input_ranges.csv', index=False)
    p = pd.read_csv(public / 'external_predictions.csv')
    joined = p.merge(ext[['row_id'] + columns], on='row_id', suffixes=('', '_input'), validate='one_to_one')
    joined.nlargest(10, 'empirical').to_csv(public / 'largest_regression_predictions.csv', index=False)
    print('Profile scenarios:', len(rows) // 3)
    print('Outside at least one checked marginal range:', int(flags.outside_any_checked_range.sum()), '/', len(flags))
    print('Missing Bq:', int(ext.bq.isna().sum()))


if __name__ == '__main__':
    main()
