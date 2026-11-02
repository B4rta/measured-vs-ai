"""Cache invalidation and metric edge cases used by the revision scripts."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from measured_vs.utils.cache import analysis_signature, protect_cache
from measured_vs.evaluation.metrics import regression_metrics_vs
from audit_validation_design import greedy_group_kfold


class CacheTests(unittest.TestCase):
    def test_cache_reuse_and_changed_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'src/measured_vs').mkdir(parents=True)
            (root/'run.py').write_text('print(1)\n')
            (root/'data.csv').write_text('x\n1\n')
            def signature():
                return analysis_signature(root, {'data': root/'data.csv'}, ['run.py'], {'trees':300})
            original = signature()
            output = root/'results'
            protect_cache(output, original)
            protect_cache(output, signature())
            (output/'predictions.csv').write_text('prediction\n1\n')
            (root/'data.csv').write_text('x\n2\n')
            with self.assertRaisesRegex(ValueError, 'new directory'):
                protect_cache(output, signature())
            for key in ['code', 'parameters', 'python', 'packages']:
                changed = copy.deepcopy(original)
                changed[key] = 'changed'
                with self.assertRaises(ValueError):
                    protect_cache(output, changed)
            (root/'data.csv').write_text('x\n1\n')
            (root/'run.py').write_text('print(2)\n')
            with self.assertRaises(ValueError):
                protect_cache(output, signature())

    def test_unregistered_predictions_are_not_silently_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'fold1').mkdir()
            (output/'fold1/outer_predictions.csv').write_text('prediction\n1\n')
            with self.assertRaisesRegex(ValueError, 'no cache signature'):
                protect_cache(output, {})
            self.assertFalse((output/'cache_signature.json').exists())

    def test_metrics_pair_rows_by_position(self):
        metrics = regression_metrics_vs(pd.Series([100.,200.], index=[1,2]),
                                        pd.Series([110.,220.], index=[3,4]))
        self.assertAlmostEqual(metrics['bias_mps'], 15.)
        self.assertAlmostEqual(metrics['rmse_mps'], np.sqrt(250.))

    def test_single_row_r2_is_undefined_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            metrics = regression_metrics_vs([100.], [110.])
        self.assertTrue(np.isnan(metrics['r2']))
        self.assertEqual(metrics['rmse_mps'], 10.)

    def test_fold_audit_matches_reference_and_tied_groups(self):
        source = pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv').group_project
        for groups in [source, pd.Series(list('aabbccddeeff'))]:
            for splits in [3,5]:
                actual = [sorted(groups.iloc[test].unique()) for _, test in
                          GroupKFold(splits).split(groups, groups=groups)]
                self.assertEqual(greedy_group_kfold(groups, splits), actual)


if __name__ == '__main__':
    unittest.main()
