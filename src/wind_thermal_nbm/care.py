"""The temperature normal behaviour models on the CARE to Compare benchmark (Gueck et al. 2024).

CARE to Compare (doi:10.5281/zenodo.15846963, CC BY-SA 4.0) holds 95 datasets from three wind farms; each is
one turbine's 10-minute SCADA with about a year of training data and a prediction section that contains either
a labelled anomaly leading up to a fault or normal behaviour. The archive is read directly from the zip.

Per dataset and method (RC model, gradient boosting):

1. every temperature channel of the farm (from ``feature_description.csv``) gets its own model, fitted on the
   first 75 % of the training section, normal status only (``status_type_id == 0``); gearbox bearings use the
   gear oil as cooling side, as on Kelmarsh;
2. the residuals are smoothed with a trailing 6-hour median over normal, producing rows (robust to the short
   sensor drop-outs in the data, e.g. -50 degC spikes) and standardised with the median and MAD (at least
   0.25 K) of the last 25 % of the training section, the calibration slice that is never used for fitting;
3. the anomaly score of a timestamp is the largest hot-side z over all channels, and a timestamp is predicted
   anomalous when that exceeds the 99.5 % quantile of the score on the calibration slice.

The point predictions are scored with :mod:`care_score`. The fleet reference does not apply: every dataset
holds one turbine only and the time stamps are anonymised per dataset.

Usage:
    meta = run_all(Path("data/care/raw/CARE_To_Compare.zip"), Path("data/care/scores"), jobs=4)
    events, scores = summarise(Path("data/care/scores"))
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .care_score import evaluate_event
from .ml import GBMModel
from .thermal import RCModel

ROOT = "CARE_To_Compare"
FARMS = ("A", "B", "C")
STEP = pd.Timedelta(minutes=10)
# operating signals per farm (sensor name of the 10-minute average), chosen from feature_description.csv
SIGNALS = {
    "A": {"power_kw": "power_30", "rotor_rpm": "sensor_52", "ambient_temp": "sensor_0", "nacelle_temp": "sensor_43"},
    "B": {"power_kw": "power_62", "rotor_rpm": "sensor_25", "ambient_temp": "sensor_8"},
    "C": {"power_kw": "power_6", "rotor_rpm": "sensor_144", "ambient_temp": "sensor_7"},
}
# temperatures that describe the surroundings rather than a component: references or left out
NOT_COMPONENTS = ("ambient", "outside", "nacelle temperature", "hub temperature", "nose cone", "platform")
FIT_FRACTION = 0.75
SMOOTH_STEPS = 36  # 6 h
MIN_SPREAD_K = 0.25  # changes below this are not actionable; avoids huge z on very steady channels
SCORE_QUANTILE = 0.995  # fixed before looking at any prediction section
QUANTILES = (0.99, SCORE_QUANTILE, 0.999)  # sensitivity check only
OPERATING_FRACTION = 0.05  # of the 99th percentile of power in the training section
METHODS = ("rc", "gbm")
log = logging.getLogger(__name__)


def _read_csv(z: zipfile.ZipFile, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)), sep=";", **kw)


def event_info(z: zipfile.ZipFile) -> pd.DataFrame:
    out = []
    for farm in FARMS:
        ev = _read_csv(z, f"{ROOT}/Wind Farm {farm}/event_info.csv").rename(columns={"asset": "asset_id"})
        ev["farm"] = farm
        out.append(ev)
    ev = pd.concat(out, ignore_index=True)
    for c in ("event_start", "event_end"):
        ev[c] = pd.to_datetime(ev[c])
    return ev


def temperature_channels(z: zipfile.ZipFile, farm: str) -> pd.DataFrame:
    """Component temperature sensors of a farm: unit degC, not one of the reference/surrounding temperatures."""
    fd = _read_csv(z, f"{ROOT}/Wind Farm {farm}/feature_description.csv", encoding="latin-1")
    unit = fd["unit"].astype(str).str.strip()
    is_temp = unit.str.endswith("C") | unit.str.lower().eq("celsius")
    desc = fd["description"].astype(str).str.strip()
    surrounding = desc.str.lower().str.contains("|".join(NOT_COMPONENTS))
    used = set(SIGNALS[farm].values())
    keep = is_temp & ~surrounding & ~fd["sensor_name"].isin(used) & fd["statistics_type"].str.contains("average")
    return pd.DataFrame({"sensor": fd.loc[keep, "sensor_name"].to_numpy(), "description": desc[keep].to_numpy()})


def gear_oil_sensor(channels: pd.DataFrame) -> str | None:
    """The gearbox oil (sump/tank) temperature, cooling side for the gearbox bearings."""
    d = channels["description"].str.lower()
    hit = channels[d.str.contains("oil") & d.str.contains("gear") & ~d.str.contains("inlet")]
    return hit["sensor"].iloc[0] if len(hit) else None


def load_dataset(z: zipfile.ZipFile, farm: str, event_id: int, sensors: list[str]) -> pd.DataFrame:
    """One dataset on a regular 10-minute grid with canonical operating columns and ``<sensor>`` targets."""
    name = f"{ROOT}/Wind Farm {farm}/datasets/{event_id}.csv"
    cols = {f"{s}_avg": s for s in sensors}
    cols.update({f"{v}_avg": k for k, v in SIGNALS[farm].items()})
    base = ["time_stamp", "train_test", "status_type_id"]
    df = _read_csv(z, name, usecols=lambda c: c in cols or c in base)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df["present"] = 1.0  # rows that exist in the file (the grid below adds empty ones)
    df = df.drop_duplicates("time_stamp").set_index("time_stamp").sort_index().rename(columns=cols)
    grid = pd.date_range(df.index[0], df.index[-1], freq=STEP)
    df = df.reindex(grid)
    df.index.name = "timestamp"
    for c in df.columns:
        if c not in ("train_test",):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["train_test"] = df["train_test"].ffill()
    return df


def anomaly_score(residuals: pd.DataFrame, usable: pd.Series, cal: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    """Largest hot-side z over channels of the trailing 6-hour median residual, standardised on the calibration
    slice."""
    keep = np.broadcast_to(usable.to_numpy()[:, None], residuals.shape)
    smooth = residuals.where(keep).rolling(SMOOTH_STEPS, min_periods=SMOOTH_STEPS // 3).median()
    ref = smooth[cal]
    med = ref.median()
    spread = (1.4826 * (ref - med).abs().median()).clip(lower=MIN_SPREAD_K)
    z = (smooth - med) / spread
    return z.max(axis=1, skipna=True), z


def score_dataset(zip_path: Path, farm: str, event_id: int, info: pd.Series, out_dir: Path) -> list[dict]:
    """Fit every channel's model for every method, save the anomaly score series and return metadata rows.

    The score file (``<farm>_<event>_<method>.parquet``) keeps per timestamp the score, the channel behind it and
    the masks needed for the evaluation, so thresholds can be varied without refitting."""
    with zipfile.ZipFile(zip_path) as z:
        channels = temperature_channels(z, farm)
        df = load_dataset(z, farm, event_id, channels["sensor"].tolist())
    oil = gear_oil_sensor(channels)
    refs = tuple(r for r in ("nacelle_temp", "ambient_temp") if r in df)
    train = (df["train_test"] == "train").to_numpy()
    normal = (df["status_type_id"] == 0).to_numpy()
    idx_train = np.flatnonzero(train)
    split = idx_train[int(FIT_FRACTION * len(idx_train))]
    fit = train & normal & (np.arange(len(df)) < split)
    cal = train & normal & (np.arange(len(df)) >= split)
    p99 = np.nanpercentile(df.loc[train, "power_kw"], 99)
    df["power_kw"] *= 2000.0 / p99  # power is rescaled for anonymisation; bring it to a 2 MW-like range
    # scored rows: the prediction section as it is in the file
    pred_rows = (df["train_test"].to_numpy() == "prediction") & (df["present"].to_numpy() == 1.0)
    # wind farm A's status comes from the failure log (two weeks before a failure are marked as downtime): it may
    # only filter the training data, so in the prediction section every producing row is used and scored
    status_ok = normal | pred_rows if farm == "A" else normal
    usable = pd.Series(status_ok & (df["power_kw"].to_numpy() > OPERATING_FRACTION * 2000.0), index=df.index)
    in_window = (df.index >= info["event_start"]) & (df.index <= info["event_end"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        residuals = {}
        for sensor, desc in zip(channels["sensor"], channels["description"]):
            if sensor not in df or np.isfinite(df[sensor].to_numpy()[fit]).sum() < 2000:
                continue
            ch_refs = refs
            if oil and sensor != oil and "bearing" in desc.lower() and "gear" in desc.lower() and oil in df:
                ch_refs = (oil, *refs)
            try:
                model = (RCModel(refs=ch_refs) if method == "rc"
                         else GBMModel(refs=ch_refs, max_iter=150, max_train_rows=20_000))
                model.fit(df, sensor, mask=fit)
            except ValueError:
                continue
            residuals[sensor] = df[sensor] - model.predict(df)
        res = pd.DataFrame(residuals, index=df.index)
        score, z = anomaly_score(res, usable, cal)
        top = pd.Series("", index=df.index, dtype=object)
        finite = score.notna().to_numpy()
        if finite.any():
            top[finite] = z[finite].idxmax(axis=1)
        pd.DataFrame({
            "score": score.astype("float32"), "top": top.astype("category"), "cal": cal, "pred": pred_rows,
            # evaluation mask: normal status on B and C; wind farm A's status is not used for the evaluation
            "status_normal": normal if farm in ("B", "C") else np.ones(len(df), dtype=bool),
            "in_window": in_window,
        }, index=df.index).to_parquet(out_dir / f"{farm}_{event_id}_{method}.parquet")
        rows.append({"farm": farm, "event_id": int(event_id), "method": method, "asset_id": int(info["asset_id"]),
                     "label": info["event_label"], "event_description": info.get("event_description", ""),
                     "channels": res.shape[1], "prediction_days": float(pred_rows.sum() / 144),
                     "event_days": float((in_window & pred_rows).sum() / 144)})
        log.info("%s %3d %-8s %-4s %3d channels scored", farm, event_id, info["event_label"], method, res.shape[1])
    return rows


def evaluate_scores(score_file: Path, label: str, quantile: float = SCORE_QUANTILE) -> dict:
    """Threshold one saved score series at the given quantile of its calibration slice and score the event."""
    s = pd.read_parquet(score_file)
    threshold = float(np.nanquantile(s.loc[s["cal"], "score"], quantile))
    p = s[s["pred"]]
    flag = (p["score"] > threshold).to_numpy()
    ev = evaluate_event(flag, p["status_normal"].to_numpy(), p["in_window"].to_numpy(), label)
    del ev["event_id"]  # the caller's metadata carries it
    flagged = flag & p["status_normal"].to_numpy()
    drivers = p.loc[flagged, "top"].astype(str).value_counts()
    return {**ev, "quantile": quantile, "threshold_z": threshold, "flagged_share": float(flagged.mean()),
            "top_channel": drivers.index[0] if len(drivers) else ""}


ZENODO_RECORD = "https://zenodo.org/api/records/15846963"


def fetch(raw_dir: Path) -> Path:
    """Download CARE_To_Compare.zip (5.5 GB) from Zenodo into ``raw_dir`` (resumable) and check its MD5."""
    import hashlib

    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "wind-thermal-nbm (open-source analysis)"
    info = next(f for f in session.get(ZENODO_RECORD, timeout=60).json()["files"] if f["key"].endswith(".zip"))
    dest = raw_dir / info["key"]
    done = dest.stat().st_size if dest.exists() else 0
    if done < info["size"]:
        log.info("downloading %s (%.1f GB, %.1f GB already here)", info["key"], info["size"] / 1e9, done / 1e9)
        headers = {"Range": f"bytes={done}-"} if done else {}
        with session.get(info["links"]["self"], headers=headers, stream=True, timeout=120) as r, dest.open("ab") as fh:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    h = hashlib.md5()
    with dest.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    algo, _, digest = info["checksum"].partition(":")
    if algo == "md5" and h.hexdigest() != digest:
        raise IOError(f"{dest.name}: MD5 mismatch, delete the file and fetch again")
    return dest


def _run_one(args) -> list[dict]:
    zip_path, farm, event_id, info, out_dir = args
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return score_dataset(zip_path, farm, event_id, info, out_dir)


def run_all(zip_path: Path, out_dir: Path, jobs: int = 1, farms=FARMS) -> pd.DataFrame:
    """Score every dataset of the benchmark into ``out_dir``; ``meta.csv`` there makes an interrupted run
    resume where it stopped."""
    out_csv = out_dir / "meta.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    with zipfile.ZipFile(zip_path) as z:
        events = event_info(z)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["farm", "event_id"])
    seen = set(zip(done["farm"], done["event_id"].astype(int)))
    todo = [(zip_path, r.farm, int(r.event_id), r) for r in events.itertuples(index=False)
            if r.farm in farms and (r.farm, int(r.event_id)) not in seen]
    todo = [(zp, f, e, pd.Series(r._asdict()), out_dir) for zp, f, e, r in todo]
    log.info("CARE: %d datasets to run, %d cached", len(todo), len(seen))
    rows = [done] if len(done) else []
    with ProcessPoolExecutor(max_workers=max(jobs, 1)) as ex:
        for res in ex.map(_run_one, todo):
            frame = pd.DataFrame(res)
            frame.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


THERMAL_WORDS = ("transformer", "gearbox", "gear oil", "bearing", "generator", "temperature", "cooling", "cooler")


def fault_family(description) -> str:
    """'thermal / drivetrain' when the logged root cause names a component a temperature sensor watches."""
    d = str(description).lower() if isinstance(description, str) else ""
    if not d:
        return ""
    return "thermal / drivetrain" if any(w in d for w in THERMAL_WORDS) else "other"


def summarise(scores_dir: Path, quantiles=(SCORE_QUANTILE,)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluated events and CARE scores (all farms and per farm) for every method and threshold quantile."""
    from .care_score import care_score

    meta = pd.read_csv(scores_dir / "meta.csv")
    events = []
    for q in quantiles:
        for r in meta.itertuples(index=False):
            ev = evaluate_scores(scores_dir / f"{r.farm}_{r.event_id}_{r.method}.parquet", r.label, q)
            events.append({**r._asdict(), **ev})
    events = pd.DataFrame(events)
    events["family"] = np.where(events["label"] == "anomaly", events["event_description"].map(fault_family), "")
    summary = []
    for (q, m), g in events.groupby(["quantile", "method"]):
        summary.append({"quantile": q, "method": m, "farm": "all", **care_score(g)})
        for farm, gf in g.groupby("farm"):
            summary.append({"quantile": q, "method": m, "farm": farm, **care_score(gf)})
    return events, pd.DataFrame(summary)
