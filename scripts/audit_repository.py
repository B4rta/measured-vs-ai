"""Check repository artifacts and saved validation results without fitting models."""
from pathlib import Path
from collections import Counter
import ast
import json
import re
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from measured_vs.evaluation.metrics import regression_metrics_vs
from measured_vs.evaluation.project_conformal import project_quantile
from measured_vs.utils.cache import analysis_signature

MODELS = ['empirical','rf','et','base_stack','specialist_stack']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(np.allclose(actual, expected, rtol=1e-9, atol=1e-9, equal_nan=True), label)


def inventory():
    command = ['git','-c',f'safe.directory={ROOT.as_posix()}', 'ls-files',
               '--cached','--others','--exclude-standard','-z']
    names = sorted(set(subprocess.check_output(command, cwd=ROOT).decode('utf8').strip('\0').split('\0')))
    counts = Counter()
    for name in names:
        path = ROOT/name
        data = path.read_bytes()
        require(not data.startswith(b'version https://git-lfs.github.com/spec/v1'),
                f'Unresolved LFS pointer: {name}; run git lfs pull')
        counts[path.suffix or '(none)'] += 1
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=name)
        elif path.suffix == '.json':
            json.loads(data.decode('utf-8-sig'))
        elif path.suffix in {'.yaml','.yml'}:
            yaml.safe_load(data.decode('utf-8-sig'))
        elif path.suffix == '.toml':
            tomllib.loads(data.decode('utf-8-sig'))
        elif path.suffix == '.csv':
            pd.read_csv(path)
        elif path.suffix == '.parquet':
            pd.read_parquet(path)
        elif path.suffix in {'.png','.jpg'}:
            with Image.open(path) as picture:
                picture.verify()
        elif path.suffix == '.svg':
            ET.fromstring(data)
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', data.decode('utf8')):
                if '://' not in target and not target.startswith('#'):
                    require((path.parent/target.split('#')[0]).exists(), f'Broken link: {name}: {target}')
    return dict(files=len(names), extensions=dict(sorted(counts.items())),
                serialized_models='LFS payloads present; historical joblib files are not unpickled or certified as revised models')


def row_identity(predictions, source):
    require(predictions.row_id.is_unique, 'Repeated prediction row IDs')
    expected = source.set_index('row_id').loc[predictions.row_id]
    close(predictions.vs_meas_mps, expected.vs_meas_mps, 'Target/row-ID mismatch')
    require(np.array_equal(predictions.group_project, expected.group_project), 'Project/row-ID mismatch')


def metrics(predictions, table, grouping='group_project'):
    require(not table.empty, 'Missing metric rows')
    for row in table.to_dict('records'):
        model = row['model']
        expected = regression_metrics_vs(predictions.vs_meas_mps, predictions[model])
        for key, value in expected.items():
            if key in row:
                close(row[key], value, f'{model}: {key} disagrees with predictions')
        for key in ['mean_project_rmse','mean_sounding_rmse']:
            if key in row:
                macro = np.mean([regression_metrics_vs(f.vs_meas_mps,f[model])['rmse_mps']
                                 for _,f in predictions.groupby(grouping)])
                close(row[key], macro, f'{model}: {key} disagrees with predictions')


def local_results(source):
    runs = 0
    reference = pd.read_csv(ROOT/'results/revised_stress/deployable_42/outer_predictions.csv')
    for path in sorted((ROOT/'results/revised_stress').glob('*/outer_predictions.csv')):
        p = pd.read_csv(path)
        require(len(p) == len(source), f'Incomplete run: {path.parent.name}')
        row_identity(p, source)
        metrics(p, pd.read_csv(path.parent/'pooled_metrics.csv'))
        project_metrics = pd.read_csv(path.parent/'project_metrics.csv')
        fold_metrics = pd.read_csv(path.parent/'fold_metrics.csv')
        selections = json.loads((path.parent/'selection.json').read_text())
        require(len(selections) == 5, 'Wrong outer-fold count')
        for fold, (a,b) in enumerate(GroupKFold(5).split(source, groups=source.group_project), 1):
            chosen = next(s for s in selections if s['fold']==fold)
            train, test = set(source.iloc[a].group_project), set(source.iloc[b].group_project)
            require(set(chosen['train_projects'])==train and set(chosen['test_projects'])==test,
                    'Saved fold allocation differs from GroupKFold')
            held = p[p.fold.eq(fold)]
            require(set(held.row_id)==set(source.iloc[b].row_id), 'Outer-fold rows differ')
            metrics(held, fold_metrics[fold_metrics.fold.eq(fold)])
        for group, f in p.groupby('group_project'):
            metrics(f, project_metrics[project_metrics.project.eq(group)])
        runs += 1
    require(runs==6,'Expected six reference runs')
    round2 = ROOT/'results/review_round2'
    scenarios=sorted(round2.glob('*/predictions.csv'))
    require(len(scenarios)==8,'Expected eight second-review scenarios')
    for path in scenarios:
        p = pd.read_csv(path)
        require(len(p)==len(source), 'Incomplete second-review scenario')
        row_identity(p,source)
        metrics(p,pd.read_csv(path.parent/'metrics.csv'))
        n_splits=15 if path.parent.name=='leave_one_project_out' else 5
        for fold,(_,b) in enumerate(GroupKFold(n_splits).split(source,groups=source.group_project),1):
            require(set(p.loc[p.fold.eq(fold),'row_id'])==set(source.iloc[b].row_id),
                    'Wrong second-review test fold')
    p = pd.read_csv(round2/'empirical_benchmark_predictions.csv')
    row_identity(p,source)
    metrics(p,pd.read_csv(round2/'empirical_benchmark_metrics.csv'))
    return reference, dict(reference_runs=runs, second_review_scenarios=8, empirical_comparators=4)


def followup(source, reference):
    out = ROOT/'results/followup_validation'
    influence = pd.read_csv(out/'training_project_influence.csv')
    jobs = sorted((out/'influence').glob('*/selection.json'))
    require(len(jobs)==60, 'Expected 60 fixed-test deletion refits')
    job_predictions = []
    for job in jobs:
        info = json.loads(job.read_text())
        p = pd.read_csv(job.parent/'predictions.csv')
        row_identity(p,source)
        held = reference[reference.fold.eq(info['fold'])]
        require(set(p.row_id)==set(held.row_id), 'Deletion changed fixed test rows')
        require(set(info['test_projects'])==set(held.group_project), 'Deletion changed test projects')
        require(info['deleted_training_project'] not in set(held.group_project), 'Deleted a test project')
        rows = influence[influence.fold.eq(info['fold']) &
                         influence.deleted_training_project.eq(info['deleted_training_project'])]
        metrics(p,rows)
        for row in rows.to_dict('records'):
            baseline = regression_metrics_vs(held.vs_meas_mps, held[row['model']])['rmse_mps']
            close(row['reference_pooled_rmse'],baseline,'Wrong fixed-test reference')
            close(row['rmse_change'],row['rmse_mps']-baseline,'Wrong deletion RMSE change')
        job_predictions.append((info,p))
    combined = pd.read_csv(out/'combined_project_deletion.csv')
    for deleted in sorted(source.group_project.unique()):
        p = reference.copy().set_index('row_id')
        for info, prediction in job_predictions:
            if info['deleted_training_project']==deleted:
                q=prediction.set_index('row_id')
                p.loc[q.index,MODELS]=q[MODELS]
        metrics(p,combined[combined.deleted_training_project.eq(deleted)])
    scenarios = sorted(out.glob('*/metrics.csv'))
    require(len(scenarios)==5,'Expected five profile scenarios')
    for path in scenarios:
        p=pd.read_csv(path.parent/'predictions.csv')
        row_identity(p,source)
        table=pd.read_csv(path)
        metrics(p,table)
        ref=reference[reference.row_id.isin(p.row_id)]
        for row in table.to_dict('records'):
            model=row['model']
            close(row['rows'],len(p),'Wrong profile row count')
            close(row['reference_same_rows_pooled_rmse'],regression_metrics_vs(ref.vs_meas_mps,ref[model])['rmse_mps'],'Wrong paired pooled reference')
            macro=np.mean([regression_metrics_vs(f.vs_meas_mps,f[model])['rmse_mps'] for _,f in ref.groupby('group_project')])
            close(row['reference_same_rows_project_rmse'],macro,'Wrong paired project reference')
    return dict(deletion_refits=len(jobs), profile_scenarios=len(scenarios))


def intervals(source):
    out=ROOT/'results/revised_intervals'
    p=pd.read_csv(out/'predictions.csv')
    require(len(p)==len(source),'Incomplete interval predictions')
    row_identity(p,source)
    coverage=pd.read_csv(out/'coverage.csv')
    splits=json.loads((out/'splits.json').read_text())
    totals={}
    for split in splits:
        groups=[set(split[k]) for k in ['training_projects','calibration_projects','test_projects']]
        require([len(g) for g in groups]==[3,9,3], 'Wrong conformal split sizes')
        require(not any(groups[a]&groups[b] for a,b in [(0,1),(0,2),(1,2)]),'Conformal project leakage')
        f=p[p.fold.eq(split['fold'])]
        require(set(f.group_project)==groups[2],'Wrong conformal test projects')
        for pct in [80,90]:
            q=project_quantile(list(split['calibration_scores'].values()),1-pct/100)
            close(f[f'lower_{pct}'],np.maximum(0,f.prediction-q),'Wrong lower interval')
            close(f[f'upper_{pct}'],f.prediction+q,'Wrong upper interval')
            covered=(f.vs_meas_mps-f.prediction).abs().le(q)
            require(np.array_equal(f[f'covered_{pct}'],covered),'Wrong coverage flags')
            for group,sub in f.groupby('group_project'):
                row=coverage[coverage.project.eq(group)&coverage.nominal.eq(pct)].iloc[0]
                close(row.row_coverage,sub[f'covered_{pct}'].mean(),'Wrong project coverage')
                close(row.mean_width,(sub[f'upper_{pct}']-sub[f'lower_{pct}']).mean(),'Wrong interval width')
                require(bool(row.whole_project_covered)==bool(sub[f'covered_{pct}'].all()),'Wrong whole-project coverage')
    for pct in [80,90]:
        totals[str(pct)]=int(coverage[coverage.nominal.eq(pct)].whole_project_covered.sum())
    return dict(test_projects=15, whole_projects_covered=totals)


def external(source):
    out=ROOT/'results/public_external'
    p=pd.read_csv(out/'external_predictions.csv')
    raw=pd.read_csv(out/'austrian_interval_table.csv').assign(row_id=lambda d:np.arange(len(d)))
    row_identity(p,raw)
    require(len(p)==2229 and p.group_cpt.nunique()==50,'Wrong external subset')
    metrics(p,pd.read_csv(out/'external_metrics.csv'),'group_cpt')
    for sounding,f in p.groupby('group_cpt'):
        table=pd.read_csv(out/'external_sounding_metrics.csv')
        metrics(f,table[table.sounding.eq(sounding)],'group_cpt')
    table=pd.read_csv(out/'external_velocity_metrics.csv')
    masks={'Vs_lt200':p.vs_meas_mps.lt(200), 'Vs_200_400':p.vs_meas_mps.ge(200)&p.vs_meas_mps.lt(400),
           'Vs_400_500':p.vs_meas_mps.ge(400)&p.vs_meas_mps.lt(500), 'Vs_ge500':p.vs_meas_mps.ge(500)}
    for label,mask in masks.items():
        metrics(p[mask],table[table.subset.eq(label)],'group_cpt')
    agefree=pd.read_csv(out/'budapest_age_free_predictions.csv')
    row_identity(agefree,source)
    metrics(agefree,pd.read_csv(out/'budapest_age_free_metrics.csv'))
    return dict(intervals=len(p), soundings=int(p.group_cpt.nunique()), basins=int(p.group_project.nunique()),
                excluded_intervals=len(pd.read_csv(out/'source_flags.csv')))


def cache_signatures():
    count=0
    paths={'training_table':ROOT/'data/cleaned/cpt_vs_labeled.csv',
           'reference_predictions':ROOT/'results/revised_stress/deployable_42/outer_predictions.csv',
           'reference_selection':ROOT/'results/revised_stress/deployable_42/selection.json'}
    for folder in ['revised_stress','review_round2','followup_validation','public_external']:
        out=ROOT/'results'/folder
        stamp=json.loads((out/'cache_signature.json').read_text())
        inputs={key:paths[key] for key in stamp['inputs'] if key!='source_zip'}
        expected=analysis_signature(ROOT,inputs,list(stamp['code']),stamp['parameters'])
        if 'source_zip' in stamp['inputs']:
            manifest=json.loads((out/'manifest.json').read_text())
            expected['inputs']['source_zip']=manifest['source_sha256']
        require(stamp==expected,f'Outdated cache signature: {folder}')
        count+=1
    return count


def main():
    report={'scope':'File integrity, stored prediction metrics, row identities and project allocations; no model refitting'}
    report['inventory']=inventory()
    source=pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv')
    parquet=pd.read_parquet(ROOT/'data/cleaned/cpt_vs_labeled.parquet')
    pd.testing.assert_frame_equal(source,parquet,check_dtype=False,check_categorical=False,
                                  rtol=1e-10,atol=1e-10)
    source['row_id']=np.arange(len(source))
    reference,report['local_validation']=local_results(source)
    report['followup']=followup(source,reference)
    report['intervals']=intervals(source)
    report['external']=external(source)
    report['current_cache_signatures']=cache_signatures()
    report['status']='passed'
    path=ROOT/'results/repository_audit.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
