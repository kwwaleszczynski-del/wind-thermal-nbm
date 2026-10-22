import zipfile

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_turbine
from wind_thermal_nbm import data, derating
from wind_thermal_nbm.ml import GBMModel
from wind_thermal_nbm.prep import clean, operating, turbine_frame

HEADER = """# This file was exported by a SCADA portal.
#
# Turbine: Kelmarsh 1
# Time zone: UTC
#
"""


def _turbine_csv() -> str:
    cols = ['# Date and time', 'Power (kW)', '"Power, Standard deviation\n (kW)"', 'Gear oil temperature (°C)',
            'Nacelle temperature (°C)', 'Rotor speed (RPM)']
    rows = ["2020-01-01 00:00:00,1500.5,10,55.1,20.0,14.0",
            "2020-01-01 00:10:00,NaN,11,-273.2,20.5,14.1"]
    return HEADER + ",".join(cols) + "\n" + "\n".join(rows) + "\n"


def _status_csv() -> str:
    return HEADER + ("Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,"
                     "IEC category\n"
                     "2020-01-01 03:00:00,2020-01-01 04:00:00,01:00:00,Warning,1700,High temp. gear bearing 1,,x,"
                     "Forced outage\n"
                     "2020-01-02 03:00:00,-,,Stop,20,Manual stop - on site,,x,Scheduled\n")


def test_read_turbine_csv_maps_columns_and_keeps_missing_as_nan():
    df = data.read_turbine_csv(_turbine_csv(), turbine=1)
    assert df.index.names == ["turbine", "timestamp"]
    assert str(df.index.get_level_values("timestamp").tz) == "UTC"
    row = df.iloc[0]
    assert row["power_kw"] == pytest.approx(1500.5) and row["gear_oil"] == pytest.approx(55.1)
    assert np.isnan(row["stator"])  # channel absent in the export
    assert np.isnan(df.iloc[1]["power_kw"])


def test_read_zip_and_plausibility_cleaning(tmp_path):
    path = tmp_path / "Kelmarsh_SCADA_2020_9999.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv", _turbine_csv())
        z.writestr("Status_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv", _status_csv())
    scada, status = data.read_zip(path)
    assert len(scada) == 2 and len(status) == 2
    assert status.loc[0, "code"] == 1700 and pd.isna(status.loc[1, "end"])
    frame = turbine_frame(scada, 1, "2020-01-01", "2020-01-01 01:00")
    assert len(frame) == 6  # complete 10-minute grid
    assert np.isnan(frame["gear_oil"].iloc[1])  # -273.2 degC sensor error removed
    assert clean(scada)["gear_oil"].isna().sum() == 1


def test_operating_mask():
    f = pd.DataFrame({"power_kw": [0.0, 50.0, 500.0, 500.0], "rotor_rpm": [0.0, 10.0, 10.0, 2.0]})
    assert operating(f).tolist() == [False, False, True, False]


def test_derating_separates_farm_wide_from_single_turbine_limitation():
    ts = pd.date_range("2020-07-01", periods=2000, freq="10min", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for t in range(1, 7):
        ws = rng.uniform(5, 9, len(ts))
        p = 2050 * ((ws - 3) / 9) ** 3
        pitch = np.zeros(len(ts))
        p_meas = p.copy()
        farm = slice(100, 130)  # everybody pitched out together: external set-point
        pitch[farm], p_meas[farm] = 8.0, p[farm] * 0.5
        if t == 2:  # turbine 2 alone, while its gear oil is at its hottest
            pitch[500:520], p_meas[500:520] = 6.0, p[500:520] * 0.6
        oil = np.full(len(ts), 50.0)
        oil[500:520] = 60.0
        rows.append(pd.DataFrame({"timestamp": ts, "turbine": t, "ws1": ws, "ws2": ws, "ambient_temp": 15.0,
                                  "power_kw": p_meas, "rotor_rpm": 12.0, "pitch": pitch,
                                  **{c: (oil if c == "gear_oil" else 40.0) for c in data.COMPONENTS.values()}}))
    out = derating.classify(pd.concat(rows, ignore_index=True))
    assert out.farm_limited.sum() == 6 * 30
    single = out[out.limited & ~out.farm_limited]
    assert set(single.turbine) == {2} and len(single) == 20
    assert single.thermal.all() and (single.hot == "gear_oil").all()
    assert single.lost_kwh.sum() == pytest.approx((0.4 * single.p_exp / 6).sum(), rel=0.05)


def test_gbm_learns_lagged_thermal_response():
    frame, _ = synthetic_turbine(days=40, seed=4)
    train = (np.arange(len(frame)) // (5 * 144)) % 2 == 0  # alternate 5-day blocks: same range of conditions
    m = GBMModel(max_iter=100, max_train_rows=5000).fit(frame, "comp", mask=train)
    r = (frame["comp"] - m.predict(frame))[~train]  # predicted on the full frame: lag features need history
    assert np.sqrt(np.nanmean(r ** 2)) < 2.0
