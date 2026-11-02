from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from review_round2 import stress_variant
from empirical_benchmarks import predict

class RoundTwoTests(unittest.TestCase):
    def setUp(self):
        self.raw=pd.read_csv(ROOT/'data/cleaned/cpt_vs_labeled.csv')

    def test_stress_variants_do_not_depend_on_targets(self):
        changed=self.raw.assign(vs_meas_mps=self.raw.vs_meas_mps*3)
        for mode in ['linear_depth','unsat_0','unsat_2','integrated_proxy']:
            a=stress_variant(self.raw,mode);b=stress_variant(changed,mode)
            np.testing.assert_allclose(a.sigma_eff_kpa,b.sigma_eff_kpa)
            self.assertTrue(a.sigma_eff_kpa.gt(0).all())

    def test_linear_depth_difference_matches_equation(self):
        a=stress_variant(self.raw,'reference');b=stress_variant(self.raw,'linear_depth')
        delta=.33*(self.raw.z_mid_m-np.log10(self.raw.z_mid_m))
        np.testing.assert_allclose(b.sigma_v_kpa-a.sigma_v_kpa,delta*self.raw.z_mid_m)

    def test_fixed_correlations_ignore_measured_velocity(self):
        a=predict(self.raw);b=predict(self.raw.assign(vs_meas_mps=1.))
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(np.isfinite(a).all().all())

    def test_integration_constant_density_hydrostatic_identity(self):
        # Equal CPT values at equal depth force the same reconstructed density.
        # Test the first point, whose density extends to ground level.
        d=stress_variant(self.raw,'integrated_proxy')
        first=d.sort_values('z_mid_m').groupby('group_cpt').head(1)
        expected=first.gamma_sat_kn_m3*first.z_mid_m-np.minimum(first.z_mid_m,first.gwl_m)
        np.testing.assert_allclose(first.sigma_v_kpa,expected)

if __name__=='__main__':unittest.main()
