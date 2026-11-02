"""Check harmonization units and interval support with a small source fixture."""
import sys, tempfile, unittest, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from public_external_validation import prepare


class PublicSourceTests(unittest.TestCase):
    def test_interval_means_units_and_source_qt(self):
        source = pd.DataFrame({
            'ID': [1, 1, 1, 1], 'test_type': ['SCPTu'] * 4,
            'basin_valley': ['Example'] * 4, 'Depth (m)': [.25, .5, .75, 1.],
            'qc (MPa)': [2., 4., 6., 8.], 'qt (MPa)': [2.1, 4.1, 6.1, 8.1],
            'fs (kPa)': [10., 20., 30., 40.], 'u2 (kPa)': [2., 4., 6., 8.],
            'u0 (kPa)': [2.4525, 4.905, 7.3575, 9.81],
            'Vs (m/s)': [None, 200., None, 300.],
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('source.csv', source.to_csv(index=False))
            result, flags = prepare(path)
        self.assertTrue(flags.empty)
        self.assertEqual(list(result.z_top_m), [0., .5])
        self.assertEqual(list(result.z_bot_m), [.5, 1.])
        self.assertAlmostEqual(result.fs_mpa.iloc[0], .015)
        self.assertAlmostEqual(result.u2_mpa.iloc[0], .003)
        self.assertAlmostEqual(result.qt_mpa.iloc[0], 3.1)
        self.assertAlmostEqual(result.rf_pct.iloc[0], .5)
        self.assertEqual(list(result.vs_meas_mps), [200., 300.])
        self.assertTrue(result.geo_age.eq('AGE_NOT_USED').all())


if __name__ == '__main__':
    unittest.main()
