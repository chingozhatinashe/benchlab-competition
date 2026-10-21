"""The regressor catalogue: every feature, its source, unit and gap rule.

docs/features.md explains the physics and the sources. `default` is a typical
value used only if the fitted climatology asset (assets/feature_climatology.json)
can't be loaded; normally a gap is filled with the training-years median.

`carry` is how stale a value may be before it counts as missing; the builders in
feature_engineering.build apply it, always from the past (never interpolated).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    source: str
    unit: str
    carry: str
    default: float
    description: str


ACE_SWEPAM = "ACE SWEPAM (SOHO daily lists + SWPC RTSW)"
ACE_MAG = "ACE MAG (SOHO daily lists + SWPC RTSW)"
ACE_EPAM = "ACE EPAM (SOHO daily lists)"
ACE_BOTH = "ACE SWEPAM + MAG"
AIA = "SDO/AIA 193 browse images"
GOES = "GOES XRS flare list (SWPC; NCEI for history)"
F107 = "F10.7 (SWPC; OMNI for history)"

_W = "window mean, needs half the hours"

FEATURES = [
    # plasma
    Feature("density_1h", "plasma", ACE_SWEPAM, "cm^-3", "3 h", 5.0, "last hourly density"),
    Feature("density_6h", "plasma", ACE_SWEPAM, "cm^-3", _W, 5.0, "6 h mean density"),
    Feature("density_24h", "plasma", ACE_SWEPAM, "cm^-3", _W, 5.0, "24 h mean density"),
    Feature("density_slope_6h", "plasma", ACE_SWEPAM, "cm^-3/h", _W, 0.0,
            "last 3 h minus previous 3 h mean density, per hour"),
    Feature("log_temperature_6h", "plasma", ACE_SWEPAM, "log10 K", _W, 4.9,
            "6 h mean proton temperature"),
    Feature("log_t_ratio_6h", "plasma", ACE_SWEPAM, "log10", _W, 0.0,
            "6 h mean of T / T_expected(V); < -0.3 marks CME ejecta"),
    Feature("pressure_1h", "plasma", ACE_SWEPAM, "nPa", "3 h", 1.5, "last hourly dynamic pressure"),
    Feature("pressure_6h", "plasma", ACE_SWEPAM, "nPa", _W, 1.5, "6 h mean dynamic pressure"),
    Feature("pressure_max_24h", "plasma", ACE_SWEPAM, "nPa", _W, 3.0,
            "largest hourly dynamic pressure in 24 h"),
    Feature("density_lag27d", "plasma", ACE_SWEPAM, "cm^-3", _W, 5.0,
            "mean density over the 72 h one solar rotation (27 d) before the forecast hours"),
    Feature("pressure_lag27d", "plasma", ACE_SWEPAM, "nPa", _W, 1.5,
            "mean dynamic pressure over the 72 h one rotation before the forecast hours"),
    Feature("density_lag13d", "plasma", ACE_SWEPAM, "cm^-3", _W, 5.0,
            "mean density over the 72 h half a rotation (13.5 d) before the forecast hours"),
    # magnetic field
    Feature("bt_1h", "field", ACE_MAG, "nT", "3 h", 5.5, "last hourly field magnitude"),
    Feature("bt_6h", "field", ACE_MAG, "nT", _W, 5.5, "6 h mean field magnitude"),
    Feature("bt_24h", "field", ACE_MAG, "nT", _W, 5.5, "24 h mean field magnitude"),
    Feature("bz_1h", "field", ACE_MAG, "nT", "3 h", 0.0, "last hourly GSM Bz"),
    Feature("bz_min_6h", "field", ACE_MAG, "nT", _W, -3.0, "lowest hourly Bz in 6 h"),
    Feature("polarity_6h", "field", ACE_MAG, "-", _W, 0.0,
            "6 h mean of -Bx/Bt: +1 away from the Sun, -1 towards (sector)"),
    Feature("polarity_lag27d", "field", ACE_MAG, "-", _W, 0.0,
            "sector polarity over the 72 h one rotation before the forecast hours"),
    Feature("b_var_6h", "field", ACE_MAG, "-", _W, 0.6,
            "6 h mean field-direction variability; low in smooth CME flux ropes"),
    # CME and shock state
    Feature("icme_6h", "cme", ACE_BOTH, "share", _W, 0.0,
            "share of the last 6 h with CME ejecta signatures (T ratio < 0.5, Bt >= 8 nT)"),
    Feature("icme_24h", "cme", ACE_BOTH, "share", _W, 0.0, "the same over 24 h"),
    Feature("hours_since_shock", "cme", ACE_BOTH, "h", "needs field data in the last 24 h",
            168.0, "hours since the last interplanetary shock passed ACE, capped at 168"),
    Feature("log_p47_1h", "cme", ACE_EPAM, "log10 flux", "3 h", 3.5,
            "last hourly EPAM 47-68 keV proton flux; rises ahead of CME shocks"),
    Feature("log_p47_slope_6h", "cme", ACE_EPAM, "log10/h", _W, 0.0,
            "trend of the 47-68 keV flux: last 3 h minus previous 3 h, per hour"),
    Feature("log_p310_1h", "cme", ACE_EPAM, "log10 flux", "3 h", 0.5,
            "last hourly EPAM 310-580 keV proton flux"),
    # coronal holes
    *[Feature(f"ch_cm_d{d}", "coronal hole", AIA, "area share", "image within 24 h", 0.03,
              f"coronal-hole share of the central-meridian strip {d} day(s) before t0")
      for d in range(6)],
    Feature("ch_east_d0", "coronal hole", AIA, "area share", "image within 24 h", 0.03,
            "coronal-hole share of the strip 20-40 deg east of the meridian"),
    Feature("ch_disk_d0", "coronal hole", AIA, "area share", "image within 24 h", 0.03,
            "coronal-hole share of the disk within 60 deg of the meridian"),
    # eruptive activity
    Feature("xray_peak_24h", "eruptive", GOES, "log10 W/m^2", "source must answer", -6.0,
            "peak X-ray flux of flares in the last 24 h (floor B1)"),
    Feature("xray_peak_72h", "eruptive", GOES, "log10 W/m^2", "source must answer", -5.7,
            "peak X-ray flux of flares in the last 72 h (floor B1)"),
    Feature("m_flares_72h", "eruptive", GOES, "count", "source must answer", 0.0,
            "M- and X-class flares in the last 72 h"),
    Feature("hours_since_m_flare", "eruptive", GOES, "h", "source must answer", 168.0,
            "hours since the last M- or X-class flare peak, capped at 168"),
    # solar activity level
    Feature("f107_prev_day", "surface", F107, "sfu", "3 days", 140.0,
            "10.7 cm radio flux of the previous UTC day"),
    Feature("f107_mean_27d", "surface", F107, "sfu", ">= 14 of 27 days", 140.0,
            "27-day mean 10.7 cm flux (one solar rotation)"),
]

NAMES = [f.name for f in FEATURES]
BY_NAME = {f.name: f for f in FEATURES}
GROUPS = list(dict.fromkeys(f.group for f in FEATURES))


def table():
    """The catalogue as rows of dicts (for docs and notebooks)."""
    return [f.__dict__.copy() for f in FEATURES]
