"""Read the temperature channels and operating signals of the open Kelmarsh SCADA data.

Each yearly ``Kelmarsh_SCADA_<year>_<id>.zip`` (Zenodo record 16807551) holds, per turbine, a 10-minute
Greenbyte export with ~460 signals and a status (event) log. Only the signals used here are kept; each year is
cached as Parquet in ``<cache_dir>``.

The raw archives are looked up in ``$KELMARSH_DATA_DIR/raw`` (so several projects can share one download) or
in ``data/raw`` of this repository; the ``fetch`` command downloads them there.

Usage:
    from wind_thermal_nbm.data import load
    scada, status = load([2019, 2020], raw_dir=Path("data/raw"), cache_dir=Path("data/processed"))
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COL = "# Date and time"
# component temperatures modelled by the normal behaviour models (Greenbyte title -> short name)
COMPONENTS = {
    "Gear oil temperature": "gear_oil",
    "Gear oil inlet temperature": "gear_inlet",
    "Generator bearing front temperature": "gen_bearing_front",
    "Generator bearing rear temperature": "gen_bearing_rear",
    "Stator temperature 1": "stator",
    "Rotor bearing temp": "rotor_bearing",
    "Front bearing temperature": "front_bearing",
    "Rear bearing temperature": "rear_bearing",
    "Transformer temperature": "transformer",
}
# operating conditions used as model inputs or for the derating analysis
CONTEXT = {
    "Power (kW)": "power_kw",
    "Rotor speed (RPM)": "rotor_rpm",
    "Generator RPM (RPM)": "gen_rpm",
    "Wind speed Sensor 1 (m/s)": "ws1",
    "Wind speed Sensor 2 (m/s)": "ws2",
    "Blade angle (pitch position) A (°)": "pitch",
    "Turbine Power setpoint (kW)": "setpoint_kw",
    "Nacelle temperature": "nacelle_temp",
    "Nacelle ambient temperature": "ambient_temp",
    "Hub temperature": "hub_temp",
}
STATUS_COLUMNS = {
    "Timestamp start": "start",
    "Timestamp end": "end",
    "Status": "status",
    "Code": "code",
    "Message": "message",
    "IEC category": "iec_category",
}
TURBINE_RE = re.compile(r"_Kelmarsh_(\d+)_")
ZENODO_RECORD = "https://zenodo.org/api/records/16807551"
log = logging.getLogger(__name__)


def column_map() -> dict[str, str]:
    """Export column name -> short name (temperatures carry a ' (°C)' suffix in the export)."""
    out = {}
    for title, short in {**COMPONENTS, **CONTEXT}.items():
        out[title if title.endswith(")") else f"{title} (°C)"] = short
    return out


def raw_dir_default(repo_data: Path) -> Path:
    env = os.environ.get("KELMARSH_DATA_DIR")
    return Path(env) / "raw" if env else repo_data / "raw"


def _header_row(text: str, marker: str) -> int:
    """Index of the column header line (exports start with a block of '#' comment lines)."""
    for i, line in enumerate(io.StringIO(text)):
        if line.startswith(marker):
            return i
        if i > 50:
            break
    raise ValueError(f"header line starting with {marker!r} not found")


def read_turbine_csv(text: str, turbine: int) -> pd.DataFrame:
    """Parse one 10-minute export; returns a frame indexed by (turbine, timestamp UTC), float32 columns."""
    cmap = column_map()
    skip = _header_row(text, TIME_COL)
    df = pd.read_csv(io.StringIO(text), skiprows=skip, usecols=lambda c: c in cmap or c == TIME_COL,
                     na_values=["NaN"])
    ts = pd.to_datetime(df[TIME_COL], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")
    out = pd.DataFrame(index=pd.DatetimeIndex(ts, name="timestamp"))
    for src, dst in cmap.items():
        out[dst] = df[src].to_numpy(dtype="float32") if src in df else np.float32(np.nan)
    out.insert(0, "turbine", np.int8(turbine))
    return out.set_index("turbine", append=True).swaplevel().sort_index()


def read_status_csv(text: str, turbine: int) -> pd.DataFrame:
    """Parse one status/event log into start/end/status/code/message/iec_category rows."""
    skip = _header_row(text, "Timestamp start")
    df = pd.read_csv(io.StringIO(text), skiprows=skip, usecols=list(STATUS_COLUMNS), dtype=str)
    df = df.rename(columns=STATUS_COLUMNS)
    for c in ("start", "end"):  # events still open at export time have end "-"
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S", errors="coerce").dt.tz_localize("UTC")
    df["code"] = pd.to_numeric(df["code"], errors="coerce")
    df.insert(0, "turbine", np.int8(turbine))
    return df


def read_zip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scada, status = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            base = name.rsplit("/", 1)[-1]
            m = TURBINE_RE.search(base)
            if not m or not base.endswith(".csv"):
                continue
            text = z.read(name).decode("utf-8-sig")
            if base.startswith("Turbine_Data"):
                scada.append(read_turbine_csv(text, int(m.group(1))))
            elif base.startswith("Status"):
                status.append(read_status_csv(text, int(m.group(1))))
            log.info("  %s", base)
    if not scada:
        raise ValueError(f"{path.name}: no turbine data files found")
    return pd.concat(scada).sort_index(), pd.concat(status, ignore_index=True)


def load(years: list[int], raw_dir: Path, cache_dir: Path,
         refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SCADA and status data for the given years, from the Parquet cache or the raw archives."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    scada, status = [], []
    for year in years:
        s_path, e_path = cache_dir / f"thermal_{year}.parquet", cache_dir / f"status_{year}.parquet"
        if s_path.exists() and e_path.exists() and not refresh:
            s, e = pd.read_parquet(s_path), pd.read_parquet(e_path)
        else:
            archives = sorted(raw_dir.glob(f"Kelmarsh_SCADA_{year}_*.zip"))
            if not archives:
                raise FileNotFoundError(f"no archive for {year} in {raw_dir}; run the 'fetch' command first")
            log.info("reading %s", archives[0].name)
            s, e = read_zip(archives[0])
            s.to_parquet(s_path)
            e.to_parquet(e_path)
            log.info("%d: %d rows, %d status events", year, len(s), len(e))
        scada.append(s)
        status.append(e)
    scada_all = pd.concat(scada)
    scada_all = scada_all[~scada_all.index.duplicated(keep="first")].sort_index()
    status_all = pd.concat(status, ignore_index=True).drop_duplicates()
    return scada_all, status_all


def fetch(years: list[int], raw_dir: Path) -> list[Path]:
    """Download the yearly SCADA archives from Zenodo (skipped when already present), checking MD5."""
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "wind-thermal-nbm (open-source analysis)"
    r = session.get(ZENODO_RECORD, timeout=60)
    r.raise_for_status()
    files = {f["key"]: f for f in r.json()["files"]}
    wanted = [k for k in files if (m := re.fullmatch(r"Kelmarsh_SCADA_(\d{4})_\d+\.zip", k)) and int(m.group(1)) in years]
    out = []
    for key in sorted(wanted):
        info, dest = files[key], raw_dir / key
        if dest.exists() and dest.stat().st_size == info["size"]:
            log.info("cached %s", key)
            out.append(dest)
            continue
        log.info("downloading %s (%.0f MB)", key, info["size"] / 1e6)
        h, tmp = hashlib.md5(), dest.with_suffix(".zip.part")
        with session.get(info["links"]["self"], stream=True, timeout=120) as resp, tmp.open("wb") as fh:
            resp.raise_for_status()
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
                h.update(chunk)
        algo, _, digest = info["checksum"].partition(":")
        if algo == "md5" and h.hexdigest() != digest:
            tmp.unlink()
            raise IOError(f"{key}: MD5 mismatch")
        tmp.replace(dest)
        out.append(dest)
    return out
