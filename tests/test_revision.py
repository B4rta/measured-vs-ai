import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from run_nested_validation import components, assert_disjoint
from measured_vs.data.features import engineer_profile_features, tree_feature_columns
from measured_vs.data.stress import rebuild_stress
from measured_vs.evaluation.project_conformal import project_quantile


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=engineer_profile_features(rebuild_stress(pd.read_csv(ROOT/"data/cleaned/cpt_vs_labeled.csv")))

    def test_calibration_rank(self):
        self.assertTrue(np.isinf(project_quantile([1,2,3,4,5],.1)))
        self.assertEqual(project_quantile(np.arange(1,10),.1),9)
        self.assertEqual(project_quantile(np.arange(1,10),.2),8)

    def test_above_water_stress_and_no_target_dependency(self):
        x=pd.DataFrame(dict(z_mid_m=[2.],gwl_m=[5.],fs_mpa=[.05],qt_mpa=[5.],u2_mpa=[0.],vs_meas_mps=[200.]))
        a=rebuild_stress(x)
        self.assertAlmostEqual(a.sigma_v_kpa.iloc[0],2*(a.gamma_sat_kn_m3.iloc[0]-1))
        x.vs_meas_mps=999
        pd.testing.assert_series_equal(a.sigma_v_kpa,rebuild_stress(x).sigma_v_kpa)

    def test_project_overlap_rejected(self):
        with self.assertRaises(ValueError): assert_disjoint(self.data,self.data.iloc[:1])

    def test_heldout_targets_and_modality_cannot_change_deployable_predictions(self):
        x=self.data
        held=x.group_project.eq("Hungaria krt")
        a=x.loc[~held].copy(); b=x.loc[held].copy()
        cols,_=tree_feature_columns(x,False)
        self.assertFalse(set(cols)&{"test_method","age_method","vs_meas_mps","log_vs_meas"})
        first=components(a,b,42,False,12)
        b.vs_meas_mps=99999.; b.log_vs_meas=-999.
        for f in [a,b]:
            f.test_method="CHANGED"; f.age_method="CHANGED"
        second=components(a,b,42,False,12)
        np.testing.assert_allclose(first,second,equal_nan=True)


if __name__=="__main__": unittest.main()
