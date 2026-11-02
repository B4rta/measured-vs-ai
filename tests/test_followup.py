from pathlib import Path
import sys,unittest
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from followup_validation import physical_windows,resampled_context
from measured_vs.data.stress import rebuild_stress

class FollowupTests(unittest.TestCase):
    def setUp(self):
        raw=pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv')
        self.raw=rebuild_stress(raw[raw.group_cpt.eq(raw.group_cpt.iloc[0])].copy())
        self.raw['row_id']=np.arange(len(self.raw))

    def test_context_does_not_use_targets(self):
        altered=self.raw.assign(vs_meas_mps=self.raw.vs_meas_mps*7,log_vs_meas=-123.)
        for builder in [physical_windows,lambda r:resampled_context(r,.5),lambda r:resampled_context(r,2.)]:
            a=builder(self.raw);b=builder(altered)
            features=[c for c in a if c not in ['vs_meas_mps','log_vs_meas']]
            pd.testing.assert_frame_equal(a[features],b[features])

    def test_physical_windows_use_depth_distance(self):
        r=self.raw.iloc[:3].copy()
        r['z_mid_m']=[1.,2.,10.];r['qc_mpa']=[1.,4.,100.]
        d=physical_windows(r)
        self.assertAlmostEqual(d.log_qc_mpa_roll3.iloc[0],np.log(2.))
        self.assertAlmostEqual(d.log_qc_mpa_roll3.iloc[2],np.log(100.))

    def test_resampling_keeps_scoring_records(self):
        d=resampled_context(self.raw,.5)
        self.assertEqual(set(d.row_id),set(self.raw.row_id))
        original=self.raw.set_index('row_id').vs_meas_mps
        np.testing.assert_allclose(d.vs_meas_mps,d.row_id.map(original))

if __name__=='__main__':unittest.main()
