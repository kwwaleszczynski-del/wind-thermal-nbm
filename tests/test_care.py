import zipfile

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_turbine
from wind_thermal_nbm import care

FEATURES = """sensor_name;statistics_type;description;unit;is_angle;is_counter
sensor_8;average,minimum;Outside temperature;\xb0C;False;False
sensor_25;average;Rotor speed;rpm;False;False
power_62;average;Active power;kW;False;False
sensor_22;average;Hub temperature;\xb0C;False;False
sensor_39;average;Gearbox oil tank temperature;\xb0C;False;False
sensor_34;average;Gearbox bearing temperature 1;\xb0C;False;False
sensor_47;average;Transformer cell temperature;\xb0C;False;False
sensor_11;average;Reactive power;kvar;False;False
"""


def _fake_zip(path, hot_kelvin: float):
    """Farm "B" layout with one anomaly dataset (id 1): 30 days training, 6 days prediction, and the transformer
    running ``hot_kelvin`` too hot inside the event window."""
    frame, _ = synthetic_turbine(days=36, seed=5)
    n = len(frame)
    ts = pd.date_range("2023-01-01", periods=n, freq="10min")
    pred = np.arange(n) >= 30 * 144
    window = (np.arange(n) >= 31 * 144) & (np.arange(n) < 35 * 144)
    oil = frame["comp"].to_numpy()
    trafo = 0.5 * frame["comp"].to_numpy() + 10 + np.where(window, hot_kelvin, 0.0)
    rows = pd.DataFrame({
        "time_stamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "asset_id": 3, "id": np.arange(n),
        "train_test": np.where(pred, "prediction", "train"), "status_type_id": 0,
        "sensor_8_avg": frame["ambient_temp"], "sensor_25_avg": frame["rotor_rpm"],
        "power_62_avg": frame["power_kw"] / 2050, "sensor_22_avg": frame["nacelle_temp"],
        "sensor_39_avg": oil, "sensor_34_avg": oil + 0.01 * frame["power_kw"], "sensor_47_avg": trafo,
        "sensor_11_avg": 0.0,
    })
    info = pd.DataFrame([{"asset_id": 3, "event_id": 1, "event_label": "anomaly",
                          "event_start": ts[31 * 144].strftime("%Y-%m-%d %H:%M:%S"),
                          "event_end": ts[35 * 144 - 1].strftime("%Y-%m-%d %H:%M:%S"),
                          "event_start_id": 31 * 144, "event_end_id": 35 * 144 - 1, "event_description": "test"}])
    root = f"{care.ROOT}/Wind Farm B"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{root}/feature_description.csv", FEATURES.encode("latin-1"))
        z.writestr(f"{root}/event_info.csv", info.to_csv(sep=";", index=False))
        z.writestr(f"{root}/datasets/1.csv", rows.to_csv(sep=";", index=False))
    return info.assign(farm="B", event_start=pd.to_datetime(info.event_start),
                       event_end=pd.to_datetime(info.event_end)).iloc[0]


def test_channel_selection_keeps_components_only(tmp_path):
    _fake_zip(tmp_path / "c.zip", 0.0)
    with zipfile.ZipFile(tmp_path / "c.zip") as z:
        ch = care.temperature_channels(z, "B")
    assert ch["sensor"].tolist() == ["sensor_39", "sensor_34", "sensor_47"]  # no outside / hub / non-temperature
    assert care.gear_oil_sensor(ch) == "sensor_39"


def test_load_dataset_regular_grid_and_names(tmp_path):
    _fake_zip(tmp_path / "c.zip", 0.0)
    with zipfile.ZipFile(tmp_path / "c.zip") as z:
        df = care.load_dataset(z, "B", 1, ["sensor_39", "sensor_47"])
    assert {"power_kw", "rotor_rpm", "ambient_temp", "sensor_47", "train_test", "present"} <= set(df.columns)
    assert (df.index[1:] - df.index[:-1] == care.STEP).all()


def test_anomaly_score_is_hot_side_and_robust_to_dropouts():
    idx = pd.date_range("2023-01-01", periods=2000, freq="10min")
    rng = np.random.default_rng(0)
    res = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)}, index=idx)
    res.iloc[1500:1503, 0] = -60.0  # a three-sample sensor drop-out
    res.iloc[1800:, 1] += 6.0  # a lasting hot shift
    cal = np.arange(2000) < 1000
    score, z = care.anomaly_score(res, pd.Series(True, index=idx), cal)
    assert score.iloc[1510] < 3  # the drop-out does not move the 6-hour median
    assert score.iloc[-1] > 4 and z.iloc[-1]["b"] == score.iloc[-1]


@pytest.mark.parametrize("hot, detected", [(8.0, True), (0.0, False)])
def test_end_to_end_detects_hot_transformer_only_when_present(tmp_path, hot, detected):
    info = _fake_zip(tmp_path / "c.zip", hot)
    meta = care.score_dataset(tmp_path / "c.zip", "B", 1, info, tmp_path / "scores")
    assert {m["method"] for m in meta} == {"rc", "gbm"}
    for m in ("rc", "gbm"):
        ev = care.evaluate_scores(tmp_path / "scores" / f"B_1_{m}.parquet", "anomaly")
        assert ev["detected"] is detected
        if detected:
            assert ev["top_channel"] == "sensor_47"


def test_fault_family_from_logged_root_cause():
    assert care.fault_family("Gearbox failure") == "thermal / drivetrain"
    assert care.fault_family("93005 : Gear oil cooler bypass valve") == "thermal / drivetrain"
    assert care.fault_family("Hydraulic group") == "other"
    assert care.fault_family("Pitch failure - defect fan on pitch motor") == "other"
    assert care.fault_family(float("nan")) == ""
