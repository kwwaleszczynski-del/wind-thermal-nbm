"""Figures for the README (``docs/img``) and the interactive dashboard (``docs/index.html``).

Usage:
    report(raw_dir=Path("data/raw"), data_dir=Path("data"), results_dir=Path("results"), docs_dir=Path("docs"))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import COMPONENTS, load  # noqa: E402
from .faults import FAULTS  # noqa: E402
from .ml import GBMModel  # noqa: E402
from .pipeline import COMPONENT_REFS, METHODS, TEST_YEARS  # noqa: E402
from .prep import turbine_frame, year_bounds  # noqa: E402
from .thermal import RCModel  # noqa: E402

METHOD_LABELS = {"rc": "RC model", "gbm": "Gradient boosting", "fleet": "Fleet median",
                 "rc_fleet": "RC + fleet", "gbm_fleet": "GBM + fleet"}
METHOD_COLORS = {"rc": "#2a78d6", "gbm": "#eb6834", "fleet": "#1baf7a", "rc_fleet": "#eda100",
                 "gbm_fleet": "#e87ba4"}
COMPONENT_LABELS = {"gear_oil": "Gear oil", "gear_inlet": "Gear oil inlet", "gen_bearing_front": "Gen. bearing front",
                    "gen_bearing_rear": "Gen. bearing rear", "stator": "Stator", "rotor_bearing": "Rotor bearing",
                    "front_bearing": "Gear bearing 1", "rear_bearing": "Gear bearing 2",
                    "transformer": "Transformer"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PRIMARY_BASELINE = "trailing"
log = logging.getLogger(__name__)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6,
    })


def fig_fit_example(scada: pd.DataFrame, out: Path, turbine: int = 3) -> None:
    """One week of measured vs simulated temperature for a passive (transformer) and a regulated (gear oil)
    component, both models trained two years earlier."""
    train = turbine_frame(scada, turbine, *year_bounds(2019))
    week = turbine_frame(scada, turbine, "2021-03-01", "2021-03-15")
    hist = turbine_frame(scada, turbine, "2021-02-25", "2021-03-15")  # lag features need history
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 2, 2]})
    axes[0].plot(week.index, week["power_kw"] / 1000, color=MUTED, lw=1.2)
    axes[0].set_ylabel("Power [MW]")
    for ax, comp in zip(axes[1:], ("transformer", "gear_oil")):
        refs = COMPONENT_REFS[comp]
        rc = RCModel(refs=refs).fit(train, comp)
        gbm = GBMModel(refs=refs).fit(train, comp)
        p_rc = rc.predict(hist).reindex(week.index)
        p_gbm = gbm.predict(hist).reindex(week.index)
        ax.plot(week.index, week[comp], color=INK, lw=1.2, label="measured")
        ax.plot(week.index, p_rc, color=METHOD_COLORS["rc"], label="RC model")
        ax.plot(week.index, p_gbm, color=METHOD_COLORS["gbm"], label="gradient boosting")
        ax.set_ylabel("°C")
        ax.set_title(f"{COMPONENT_LABELS[comp]}, turbine {turbine} (RC time constant {rc.tau_h:.1f} h)", loc="left")
    axes[1].legend(loc="upper left", ncols=3, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_residual_spread(spread: pd.DataFrame, out: Path) -> None:
    """Median daily residual spread per component and method (calibration years)."""
    s = spread.groupby(["component", "method"]).daily_sd_K.median().unstack("method")
    s = s.loc[list(COMPONENTS.values()), list(METHODS)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    n = len(METHODS)
    w = 0.8 / n
    x = np.arange(len(s))
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - (n - 1) / 2) * w, s[m], width=w * 0.88, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_xticks(x, [COMPONENT_LABELS[c] for c in s.index], rotation=20, ha="right")
    ax.set_ylabel("Daily residual spread [K]")
    ax.set_title("Noise of the daily residual (robust SD, median over turbines and calibration years)", loc="left")
    ax.legend(ncols=5, loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_injections(summary: pd.DataFrame, out: Path) -> None:
    """Detection rate (bars) and median days to detection (labels) per fault and method, CUSUM chart."""
    s = summary[(summary.chart == "cusum") & (summary.baseline == PRIMARY_BASELINE)]
    faults = list(FAULTS)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    n = len(METHODS)
    w = 0.8 / n
    x = np.arange(len(faults))
    for i, m in enumerate(METHODS):
        d = s[s.method == m].set_index("fault").reindex(faults)
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, d.detected_pct, width=w * 0.88, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for xi, (pct, days) in zip(xs, zip(d.detected_pct, d.median_days)):
            if np.isfinite(days) and pct >= 15:
                ax.text(xi, pct + 1.5, f"{days:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    chance = s[s.fault == "none_hot"].detected_pct.mean()
    ax.axhline(chance, color=MUTED, lw=1)
    short = {"drift_2": "drift +2 °C", "drift_5": "drift +5 °C", "cooling_25": "cooling R +25 %",
             "step_3": "step +3 °C", "stuck": "stuck sensor"}
    ax.set_xticks(x, [short[f] for f in faults], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Detected within 60 days [%]")
    ax.set_title("Injected faults: detection rate (labels: median days to detection), CUSUM\n"
                 f"grey line: chance of a hot-side alarm in the same windows without a fault ({chance:.0f} %)",
                 loc="left")
    ax.legend(ncols=5, loc="upper left", fontsize=9, bbox_to_anchor=(0, -0.12))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_tradeoff(summary: pd.DataFrame, fa: pd.DataFrame, out: Path, fault: str = "drift_2") -> None:
    """False alarms on the unmodified test years against detection of the smallest injected fault."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in METHODS:
        pts = []
        for b in ("fixed", "trailing"):
            det = summary[(summary.method == m) & (summary.baseline == b) & (summary.chart == "cusum")
                          & (summary.fault == fault)].detected_pct.iloc[0]
            f = fa[(fa.method == m) & (fa.baseline == b) & (fa.chart == "cusum")].per_turbine_year.iloc[0]
            pts.append((f, det))
        (x0, y0), (x1, y1) = pts
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "color": METHOD_COLORS[m], "lw": 1.4})
        ax.scatter([x0], [y0], s=60, facecolor="#fcfcfb", edgecolor=METHOD_COLORS[m], lw=2, zorder=3)
        ax.scatter([x1], [y1], s=70, color=METHOD_COLORS[m], edgecolor="#fcfcfb", lw=2, zorder=3,
                   label=METHOD_LABELS[m])
        ax.annotate(METHOD_LABELS[m], (x1, y1), xytext=(-10, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=8, color=INK2)
    ax.set_xlabel("False alarms per turbine-year (9 channels, unmodified 2019-2021)")
    ax.set_ylabel(f"Detected within 60 days [%]: {FAULTS[fault]}")
    ax.set_title("Trade-off per method: hollow = fixed baseline, filled = trailing baseline", loc="left")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_event(z: pd.DataFrame, alarms: pd.DataFrame, events: pd.DataFrame, out: Path, turbine: int,
              component: str, start: str, end: str) -> None:
    """Daily z of one channel for every method around logged events, with CUSUM alarms."""
    zz = z[(z.index >= pd.Timestamp(start, tz="UTC")) & (z.index < pd.Timestamp(end, tz="UTC"))]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, sharey=True)
    for ax, m in zip(axes, ("rc_fleet", "gbm", "fleet")):
        col = f"{m}|{component}|{turbine}"
        ax.plot(zz.index, zz[col], color=METHOD_COLORS[m], lw=1.4)
        a = alarms[(alarms.channel == col) & (alarms.chart == "cusum") & (alarms.direction > 0)]
        a = a[(a.day >= zz.index[0]) & (a.day <= zz.index[-1])]
        if len(a):
            ax.scatter(a.day, zz[col].reindex(a.day), s=36, color=METHOD_COLORS[m], edgecolor="#fcfcfb", lw=1.5,
                       zorder=3, label="CUSUM alarm (hot)")
            ax.legend(loc="upper left", fontsize=9)
        ev = events[(events.turbine == turbine) & (events.method == m)
                    & (events.group == "gear bearing high temperature")]
        for _, e in ev.iterrows():
            if zz.index[0] <= e.start <= zz.index[-1]:
                ax.axvline(e.start, color="#d03b3b", lw=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-8, 19)
        ax.set_ylabel("z")
        ax.set_title(f"{METHOD_LABELS[m]}", loc="left")
    fig.suptitle(f"Turbine {turbine}, {COMPONENT_LABELS[component]}: daily standardised residual (trailing "
                 "baseline); red lines = logged 'High temp. gear bearing' warnings", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_derating(by_ambient: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(by_ambient))
    ax.bar(x - 0.2, by_ambient.farm_wide_pct, width=0.36, color="#6250d6", label="farm-wide (external set-point)")
    ax.bar(x + 0.2, by_ambient.single_turbine_pct, width=0.36, color="#eb6834", label="single turbine")
    edges = [b.strip("(]").split(", ") for b in by_ambient.ambient_band.astype(str)]
    ax.set_xticks(x, [f"{lo} to {hi} °C" for lo, hi in edges], fontsize=9)
    ax.set_ylabel("Share of running time [%]")
    ax.set_title("Power limited below rated wind (pitched out, < 93 % rated), by ambient temperature", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_rc_trend(by_year: pd.DataFrame, out: Path, highlight=(5, 6)) -> None:
    """Rise at rated power per calendar year and turbine: the physical health trend of the RC model."""
    comps = ("gen_bearing_rear", "stator", "front_bearing", "transformer")
    fig, axes = plt.subplots(1, len(comps), figsize=(11, 3.6), sharex=True)
    colors = {highlight[0]: "#2a78d6", highlight[1]: "#eb6834"}
    for ax, comp in zip(axes, comps):
        d = by_year[by_year.component == comp].pivot(index="year", columns="turbine", values="rise_rated_K")
        for t in d.columns:
            if t in colors:
                continue
            ax.plot(d.index, d[t], color="#c3c2b7", lw=1.4, label="other turbines" if t == d.columns[0] else None)
        for t in highlight:
            ax.plot(d.index, d[t], color=colors[t], lw=2, marker="o", ms=5, mec="#fcfcfb", mew=1.5,
                    label=f"T{t}")
        ax.set_title(COMPONENT_LABELS[comp], loc="left")
        ax.set_xticks(list(d.index), [str(y)[2:] for y in d.index])
    axes[0].set_ylabel("Rise at rated power [K]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("RC model fitted per calendar year: temperature rise above the cooling side at rated power",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def monthly_heat(z: pd.DataFrame) -> dict:
    """Monthly mean z per method and channel (rows: turbine x component)."""
    months = z.index.strftime("%Y-%m")
    g = z.groupby(months).mean()
    out = {}
    for m in METHODS:
        rows = []
        for t in range(1, 7):
            for c in COMPONENTS.values():
                rows.append([None if not np.isfinite(v) else round(float(v), 2) for v in g[f"{m}|{c}|{t}"]])
        out[m] = rows
    return {"months": list(g.index), "values": out}


def dashboard(z: pd.DataFrame, alarms: pd.DataFrame, events: pd.DataFrame, params: pd.DataFrame,
              out: Path, template: Path) -> None:
    channels = [(t, c) for t in range(1, 7) for c in COMPONENTS.values()]
    daily = {m: [[None if not np.isfinite(v) else round(float(v), 1) for v in z[f"{m}|{c}|{t}"]]
                 for t, c in channels] for m in METHODS}
    a = alarms[alarms.chart == "cusum"]
    ch_index = {f"{t}|{c}": i for i, (t, c) in enumerate(channels)}
    day0 = z.index[0]
    alarm_list = {m: [[ch_index[f"{r.turbine}|{r.component}"], int(round((r.day - day0) / pd.Timedelta(days=1))),
                       int(r.direction)] for r in a[a.method == m].itertuples()] for m in METHODS}
    ev = events.drop_duplicates(["group", "turbine", "start"])
    event_list = [{"turbine": int(r.turbine), "day": r.start.strftime("%Y-%m-%d"), "group": r.group,
                   "messages": r.messages, "campaign": bool(r.campaign)} for r in ev.itertuples()]
    p = params[params.test_year == max(TEST_YEARS)]
    rc_params = {f"{r.turbine}|{r.component}": {"tau": round(r.tau_h, 2), "rise": round(r.rise_rated_K, 1),
                                                 "rmse": round(r.rmse_K, 2)} for r in p.itertuples()}
    payload = {
        "methods": [{"key": m, "label": METHOD_LABELS[m], "color": METHOD_COLORS[m]} for m in METHODS],
        "components": [{"key": c, "label": COMPONENT_LABELS[c]} for c in COMPONENTS.values()],
        "channels": [[t, c] for t, c in channels],
        "start": day0.strftime("%Y-%m-%d"),
        "heat": monthly_heat(z),
        "daily": daily,
        "alarms": alarm_list,
        "events": event_list,
        "rc_params": rc_params,
    }
    html = template.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    out.write_text(html, encoding="utf-8")
    log.info("dashboard %s (%.1f MB)", out, out.stat().st_size / 1e6)


def report(raw_dir: Path, data_dir: Path, results_dir: Path, docs_dir: Path) -> None:
    _style()
    img = docs_dir / "img"
    img.mkdir(parents=True, exist_ok=True)
    models = data_dir / "models"
    z = pd.read_parquet(models / "daily_z.parquet").xs(PRIMARY_BASELINE, level="baseline")
    alarms = pd.read_csv(results_dir / "alarms.csv", parse_dates=["day"])
    alarms = alarms[alarms.baseline == PRIMARY_BASELINE]
    events = pd.read_csv(results_dir / "events.csv", parse_dates=["start", "end"])
    events = events[events.baseline == PRIMARY_BASELINE]
    spread = pd.read_csv(results_dir / "residual_spread.csv")
    summary = pd.read_csv(results_dir / "injection_summary.csv")
    params = pd.read_csv(results_dir / "rc_params.csv")
    scada, _ = load([2019, 2021], raw_dir, data_dir / "processed")
    fig_fit_example(scada, img / "fit_example.png")
    fig_residual_spread(spread, img / "residual_spread.png")
    fig_injections(summary, img / "injections.png")
    fig_tradeoff(summary, pd.read_csv(results_dir / "false_alarms.csv"), img / "tradeoff.png")
    fig_event(z, alarms, events, img / "event_t6_gear_bearing.png", turbine=6, component="front_bearing",
              start="2020-09-01", end="2021-04-30")
    fig_derating(pd.read_csv(results_dir / "derating_by_ambient.csv"), img / "derating.png")
    fig_rc_trend(pd.read_csv(results_dir / "rc_by_year.csv"), img / "rc_trend.png")
    dashboard(z, alarms, events, params, docs_dir / "index.html", Path(__file__).with_name("dashboard.html"))

