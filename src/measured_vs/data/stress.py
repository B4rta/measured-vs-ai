"""CPT-only stress proxies, with explicit units and no measured-Vs inputs."""
import numpy as np
import pandas as pd


def rebuild_stress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z = pd.to_numeric(out.z_mid_m)
    gwl = pd.to_numeric(out.gwl_m)
    fs = pd.to_numeric(out.fs_mpa) * 1000
    qt = pd.to_numeric(out.qt_mpa) * 1000
    # Source workbook CPT_SCPT_MASW!S5: LOG means base 10, including depth.
    gamma = 11.46 + .33*np.log10(z.where(z > 0)) + 3.1*np.log10(fs.where(fs > 0)) + .7*np.log10(qt.where(qt > 0))
    out["gamma_sat_kn_m3"] = gamma
    out["gamma_unsat_kn_m3"] = gamma - 1.
    above = np.minimum(z, np.maximum(gwl, 0))
    below = np.maximum(z-gwl, 0)
    out["z_above_gwl_m"] = np.maximum(gwl-z, 0)
    out["z_below_gwl_m"] = below
    # Homogeneous-column approximation at each interval, not measured stresses.
    out["sigma_v_kpa"] = (gamma-1)*above + gamma*below
    out["u0_kpa"] = 9.81*below
    out["sigma_eff_kpa"] = out.sigma_v_kpa - out.u0_kpa
    denominator = qt-out.sigma_v_kpa
    out["bq"] = (pd.to_numeric(out.u2_mpa)*1000-out.u0_kpa)/denominator.where(denominator > 0)
    return out
