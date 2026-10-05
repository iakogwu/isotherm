"""Phase 2 (forecast skill) and Phase 3 (paper-trading backtest) reports.

    python -m isotherm evaluate

Split discipline:
  DEV         target dates before HOLDOUT_START. The blend is fit on the earlier part of DEV
              and its trading threshold chosen on the later part (BLEND_TUNE_FRAC), so the
              threshold is never picked on the same contracts the weights were fit on.
  VALIDATION  HOLDOUT_START onward. Reported at DEV-chosen settings, but it has been looked at
              while the model was built (2026-10-04 audit), so it is not a clean holdout.
  FORWARD     paper_predictions logged live by the scheduler. The only clean test.

Uncertainty is clustered by calendar date: brackets of one event are mutually exclusive and
weather errors are correlated across cities on the same day, so contracts are not independent.
"""
import math
import re
from datetime import date

import numpy as np
import pandas as pd

from . import config, model

HOLDOUT_START = "2026-04-01"
THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15)
CONTRACTS_PER_ORDER = 10
PRICE_RANGE = (0.03, 0.97)       # ignore quotes at the extremes (no real liquidity there)
QUOTE_MAX_AGE_H = 2
REPORT_DIR = config.DATA_DIR / "reports"
CLI_RX = re.compile(r"\((CLI\w{3})\)")


def taker_fee(price, contracts=CONTRACTS_PER_ORDER, multiplier=1.0):
    """Kalshi general fee: ceil(0.07 * mult * C * P * (1-P)) dollars per order, per contract here.
    Assumption: series fee_type 'quadratic' with fee_multiplier 1. Verify against Kalshi's schedule."""
    price = np.asarray(price, float)
    total = np.ceil(np.round(0.07 * multiplier * contracts * price * (1 - price) * 100, 6)) / 100
    return total / contracts


def split_of(d):
    return np.where(pd.Series(d).astype(str) < HOLDOUT_START, "dev", "validation")


# ------------------------------------------------------------------ data assembly

def predictions(conn, params=None) -> pd.DataFrame:
    frames = []
    for hz in ("open", "eve"):
        ds = model.build_dataset(conn, hz)
        if not ds.empty:
            frames.append(model.fit_predict(ds, params))
    pr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not pr.empty:
        pr["split"] = split_of(pr["target_date"])
    return pr


def contracts(conn, preds: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_sql_query(
        """SELECT m.ticker, m.event_ticker, m.target_date, m.strike_type, m.floor_strike, m.cap_strike,
                  m.result, m.rules_primary, s.station_id, s.kind, st.cli_code
           FROM markets m JOIN series s USING(series_ticker) JOIN stations st USING(station_id)
           WHERE m.result IN ('yes','no')""", conn)
    # drop contracts whose rules name a different station than the series does today
    code = m["rules_primary"].fillna("").str.extract(CLI_RX)[0]
    m = m[code.isna() | (code == m["cli_code"])].drop(columns=["rules_primary", "cli_code"])
    b = [model.bracket_bounds(t, f, c) for t, f, c in zip(m["strike_type"], m["floor_strike"], m["cap_strike"])]
    m = m[[x is not None for x in b]].copy()
    b = [x for x in b if x is not None]
    m["lo"] = [x[0] for x in b]
    m["hi"] = [x[1] for x in b]
    m["y"] = (m["result"] == "yes").astype(int)

    c = m.merge(preds, on=["station_id", "kind", "target_date"], how="inner")
    c["p_model"] = model.prob_in(c["mu"], c["sigma"], c["lo"], c["hi"])
    nb_ok = c["nbs"].notna()
    c["p_nbs"] = np.nan
    c.loc[nb_ok, "p_nbs"] = model.prob_in(
        c.loc[nb_ok, "nbs"], np.maximum(c.loc[nb_ok, "nbs_sd"].fillna(2.0), model.DEFAULTS["sigma_floor"]),
        c.loc[nb_ok, "lo"], c.loc[nb_ok, "hi"])

    # market quote: last hourly candle that closed at or before the decision time
    candles = pd.read_sql_query("SELECT ticker, end_ts, yes_bid, yes_ask FROM market_candles", conn)
    c["decision_ts"] = (c["decision"].astype("int64") // 10**9).astype("int64")
    if candles.empty:
        c["yes_bid"] = c["yes_ask"] = np.nan
        return c
    candles = candles.sort_values("end_ts")
    c = c.sort_values("decision_ts")
    c = pd.merge_asof(c, candles, left_on="decision_ts", right_on="end_ts", by="ticker",
                      direction="backward", tolerance=QUOTE_MAX_AGE_H * 3600)
    return c


# ------------------------------------------------------------------ scoring helpers

def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _logloss(p, y):
    p = np.clip(p, 0.01, 0.99)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fmt(df):
    return df.to_string(index=False, float_format=lambda x: f"{x:.4f}")


def phase2(preds: pd.DataFrame, con: pd.DataFrame) -> str:
    out = ["PHASE 2 - forecast skill (lower is better). Model vs NBS (NWS's own blend).", ""]
    p = preds.dropna(subset=["obs"]).copy()
    # score the distribution that is traded (Student-t, scaled sigma), for model and NBS alike
    p["crps_model"] = model.crps_traded(p["mu"], p["sigma"], p["obs"])
    has = p["nbs"].notna()
    p["crps_nbs"] = np.nan
    p.loc[has, "crps_nbs"] = model.crps_traded(
        p.loc[has, "nbs"], np.maximum(p.loc[has, "nbs_sd"].fillna(2.0), 0.7), p.loc[has, "obs"])
    p["ae_model"] = (p["mu"] - p["obs"]).abs()
    p["ae_nbs"] = (p["nbs"] - p["obs"]).abs()
    z = (p["obs"] - p["mu"]) / p["sigma"]
    p["in80"] = (z.abs() <= model.central_z(0.8)).astype(float)
    pb = p[has]
    g = pb.groupby(["split", "horizon", "kind"]).agg(
        days=("obs", "size"), mae_model=("ae_model", "mean"), mae_nbs=("ae_nbs", "mean"),
        crps_model=("crps_model", "mean"), crps_nbs=("crps_nbs", "mean"),
        cover80=("in80", "mean")).reset_index()
    g["crps_skill_vs_nbs"] = 1 - g["crps_model"] / g["crps_nbs"]
    out += ["Continuous (rows where NBS exists). cover80 should be ~0.80 if calibrated:", _fmt(g), ""]

    q = con.dropna(subset=["yes_bid", "yes_ask", "p_nbs"])
    q = q[(q["yes_ask"] > q["yes_bid"]) & (q["yes_bid"] > 0) & (q["yes_ask"] < 1)].copy()
    q["p_mkt"] = (q["yes_bid"] + q["yes_ask"]) / 2
    rows = []
    for (s, hz), gq in q.groupby(["split", "horizon"]):
        y = gq["y"].to_numpy()
        rows.append({"split": s, "horizon": hz, "contracts": len(gq),
                     "brier_model": _brier(gq["p_model"], y), "brier_nbs": _brier(gq["p_nbs"], y),
                     "brier_market": _brier(gq["p_mkt"], y),
                     "logloss_model": _logloss(gq["p_model"], y), "logloss_nbs": _logloss(gq["p_nbs"], y),
                     "logloss_market": _logloss(gq["p_mkt"], y)})
    out += ["Per-contract, where a two-sided market quote existed at decision time.",
            "If brier_market < brier_model, the market already knows more than the model:",
            _fmt(pd.DataFrame(rows)), ""]

    cal = q.copy()
    cal["bin"] = pd.cut(cal["p_model"], [0, .05, .15, .3, .5, .7, .85, .95, 1.0], include_lowest=True)
    ct = cal.groupby(["split", "bin"], observed=True).agg(
        n=("y", "size"), mean_p=("p_model", "mean"), freq=("y", "mean"),
        mean_mkt=("p_mkt", "mean")).reset_index()
    out += ["Calibration of model probabilities (freq should track mean_p):", _fmt(ct), ""]

    st = pb[pb["split"] == "validation"].groupby(["station_id", "kind"]).agg(
        days=("obs", "size"), crps_model=("crps_model", "mean"), crps_nbs=("crps_nbs", "mean")).reset_index()
    st["skill"] = 1 - st["crps_model"] / st["crps_nbs"]
    out += ["VALIDATION skill vs NBS by station (positive = model better):",
            _fmt(st.sort_values("skill", ascending=False)), ""]
    return "\n".join(out)


# ------------------------------------------------------------------ Phase 3

def _logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def quoted(con: pd.DataFrame) -> pd.DataFrame:
    """Contracts with a usable two-sided quote at decision time, plus the market mid."""
    q = con.dropna(subset=["yes_bid", "yes_ask"])
    q = q[(q["yes_ask"] > q["yes_bid"]) & (q["yes_bid"] > 0) & (q["yes_ask"] < 1)].copy()
    q["p_mkt"] = (q["yes_bid"] + q["yes_ask"]) / 2
    return q


def fit_blend(q: pd.DataFrame):
    """Logistic recalibration: logit p = b0 + b1 logit(market mid) + b2 logit(model).
    Start from the market, let the model nudge it only as much as history justifies."""
    from scipy.optimize import minimize
    X = np.column_stack([np.ones(len(q)), _logit(q["p_mkt"]), _logit(q["p_model"])])
    y = q["y"].to_numpy()
    nll = lambda b: np.sum(np.logaddexp(0, -(2 * y - 1) * (X @ b)))
    return minimize(nll, np.array([0.0, 1.0, 0.0]), method="BFGS").x


def apply_blend(q: pd.DataFrame, b) -> np.ndarray:
    z = b[0] + b[1] * _logit(q["p_mkt"]) + b[2] * _logit(q["p_model"])
    return 1 / (1 + np.exp(-z))


def trades(con: pd.DataFrame, threshold: float, col: str = "p_model") -> pd.DataFrame:
    q = con.dropna(subset=["yes_bid", "yes_ask", col])
    q = q[(q["yes_ask"] > q["yes_bid"])].copy()
    ask, bid, p = q["yes_ask"].to_numpy(), q["yes_bid"].to_numpy(), q[col].to_numpy()
    no_price = 1 - bid
    edge_yes = p - ask - taker_fee(ask)
    edge_no = (1 - p) - no_price - taker_fee(no_price)
    ok_yes = (ask >= PRICE_RANGE[0]) & (ask <= PRICE_RANGE[1])
    ok_no = (no_price >= PRICE_RANGE[0]) & (no_price <= PRICE_RANGE[1])
    edge_yes = np.where(ok_yes, edge_yes, -np.inf)
    edge_no = np.where(ok_no, edge_no, -np.inf)
    take_yes = (edge_yes >= edge_no) & (edge_yes > threshold)
    take_no = (edge_no > edge_yes) & (edge_no > threshold)
    q["side"] = np.where(take_yes, "yes", np.where(take_no, "no", None))
    q = q[q["side"].notna()].copy()
    yes = q["side"] == "yes"
    q["price"] = np.where(yes, q["yes_ask"], 1 - q["yes_bid"])
    q["fee"] = taker_fee(q["price"])
    q["exp_edge"] = np.where(yes, q[col], 1 - q[col]) - q["price"] - q["fee"]
    q["won"] = np.where(yes, q["y"] == 1, q["y"] == 0).astype(int)
    q["pnl"] = q["won"] - q["price"] - q["fee"]
    return q


def _date_boot(df: pd.DataFrame, col: str, n_boot=2000, seed=7):
    """90% CI for the mean of `col`, resampling whole calendar dates: one day's weather moves
    every city and every bracket together, so resampling contracts would overstate certainty."""
    if df.empty:
        return (np.nan, np.nan)
    g = df.groupby("target_date")[col].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return tuple(np.percentile(means, [5, 95]))


def _boot_ci(t: pd.DataFrame, n_boot=2000, seed=7):
    return _date_boot(t, "pnl", n_boot, seed)


def _summ(t):
    lo, hi = _boot_ci(t)
    return {"trades": len(t), "win_rate": t["won"].mean() if len(t) else np.nan,
            "avg_price": t["price"].mean() if len(t) else np.nan,
            "exp_edge": t["exp_edge"].mean() if len(t) else np.nan,
            "pnl_per_contract": t["pnl"].mean() if len(t) else np.nan,
            "ci90_lo": lo, "ci90_hi": hi, "total_pnl_$1": t["pnl"].sum()}


BLEND_FILE = REPORT_DIR / "blend.json"
MIN_BLEND_FIT = 2000
BLEND_TUNE_FRAC = 0.3   # latest 30% of DEV dates: threshold choice only, never blend fitting


def config_fingerprint() -> dict:
    """Every setting that changes what the frozen blend means. paper.py refuses a blend.json
    whose hash doesn't match the running code, so a tweak can't silently reuse old weights."""
    return {"model_version": model.VERSION, "params": model.DEFAULTS, "tail_df": model.TAIL_DF,
            "sigma_scale": model.SIGMA_SCALE, "holdout_start": HOLDOUT_START,
            "contracts_per_order": CONTRACTS_PER_ORDER, "price_range": list(PRICE_RANGE),
            "quote_max_age_h": QUOTE_MAX_AGE_H, "min_blend_fit": MIN_BLEND_FIT,
            "blend_tune_frac": BLEND_TUNE_FRAC,
            "thresholds_swept": list(THRESHOLDS)}


def config_hash() -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(config_fingerprint(), sort_keys=True).encode()).hexdigest()[:12]


def _git_commit() -> str:
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "isotherm"], cwd=config.ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _freeze_blend(coefs, chosen) -> str:
    """Write blend.json once, read-only. Later runs save a dated candidate instead, so the
    weights paper trading uses can't be replaced by re-running evaluate."""
    import json
    import os
    import stat
    from datetime import datetime, timezone
    doc = {"model_version": model.VERSION, "config_hash": config_hash(), "config": config_fingerprint(),
           "git_commit": _git_commit(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fit_on": "earlier DEV dates (see coefs fit_dates); threshold on later DEV dates", "coefs": coefs, "thresholds": chosen}
    text = json.dumps(doc, indent=2)
    if BLEND_FILE.exists():
        cand = REPORT_DIR / f"blend_candidate_{doc['created_at'][:19].replace(':', '')}.json"
        cand.write_text(text)
        return (f"blend.json is already frozen and was NOT changed. This run's fit is in {cand.name}. "
                "To replace the frozen blend, delete blend.json deliberately and re-run.")
    BLEND_FILE.write_text(text)
    os.chmod(BLEND_FILE, stat.S_IREAD)
    return f"Froze blend.json (config {doc['config_hash']}, commit {doc['git_commit']})."


def dev_parts(con: pd.DataFrame) -> pd.Series:
    """'fit' for the earlier DEV dates, 'tune' for the latest BLEND_TUNE_FRAC of them (by dates
    that have quotes), None outside DEV."""
    td = con["target_date"].astype(str)
    dev = con["split"] == "dev"
    dates = np.sort(td[dev & con["yes_ask"].notna()].unique())
    if len(dates) == 0:
        return pd.Series(None, index=con.index, dtype=object)
    cut = dates[int(len(dates) * (1 - BLEND_TUNE_FRAC))]
    return pd.Series(np.where(~dev, None, np.where(td < cut, "fit", "tune")), index=con.index)


def _coef_ci(q: pd.DataFrame, n_boot=200, seed=11):
    """90% CI for each blend coefficient, refitting on calendar dates resampled with replacement."""
    groups = list(q.groupby("target_date").indices.values())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        bs.append(fit_blend(q.iloc[np.concatenate([groups[i] for i in pick])]))
    return np.percentile(np.array(bs), [5, 95], axis=0)


def add_blend(con: pd.DataFrame):
    """Fit the market+model blend per horizon on the earlier DEV dates only; add p_blend to
    every quoted row (the later DEV dates and VALIDATION are out of sample for it)."""
    con = con.copy()
    con["dev_part"] = dev_parts(con)
    q = quoted(con)
    con["p_mkt"] = np.nan
    con.loc[q.index, "p_mkt"] = q["p_mkt"]
    con["p_blend"] = np.nan
    coefs = {}
    for hz in ("open", "eve"):
        fit = q[(q["dev_part"] == "fit") & (q["horizon"] == hz)]
        if len(fit) < MIN_BLEND_FIT:
            continue
        b = fit_blend(fit)
        lo, hi = _coef_ci(fit)
        coefs[hz] = {"b0": b[0], "b_market": b[1], "b_model": b[2],
                     "b_model_ci90": [lo[2], hi[2]], "b_market_ci90": [lo[1], hi[1]],
                     "n_contracts": len(fit), "n_events": int(fit["event_ticker"].nunique()),
                     "n_dates": int(fit["target_date"].nunique()),
                     "fit_dates": [str(fit["target_date"].min()), str(fit["target_date"].max())]}
        rows = q[q["horizon"] == hz]
        con.loc[rows.index, "p_blend"] = apply_blend(rows, b)
    return con, coefs


def event_scores(con: pd.DataFrame, cols) -> pd.DataFrame:
    """One row per (event, horizon) whose every bracket had a quote and exactly one settled YES.
    Each column's bracket probabilities are renormalised to sum to 1 across the event, then
    scored as one multiclass forecast: log score (-log p of the winner) and multiclass Brier."""
    q = con.loc[quoted(con).index].dropna(subset=list(cols))   # keep the columns as given
    n_all = con.groupby(["event_ticker", "horizon"]).size().rename("n_all")
    q = q.join(n_all, on=["event_ticker", "horizon"])
    g = q.groupby(["event_ticker", "horizon"])
    q = q[(g["ticker"].transform("size") == q["n_all"]) & (g["y"].transform("sum") == 1)].copy()
    if q.empty:
        return pd.DataFrame()
    out = q.groupby(["event_ticker", "horizon"]).agg(
        target_date=("target_date", "first"), split=("split", "first"),
        dev_part=("dev_part", "first")).reset_index()
    for c in cols:
        pn = q[c] / q.groupby(["event_ticker", "horizon"])[c].transform("sum")
        q["_ls"] = np.where(q["y"] == 1, -np.log(np.clip(pn, 1e-4, 1)), 0.0)
        q["_br"] = (pn - q["y"]) ** 2
        agg = q.groupby(["event_ticker", "horizon"]).agg(ls=("_ls", "sum"), br=("_br", "sum")).reset_index()
        out = out.merge(agg.rename(columns={"ls": f"ls_{c}", "br": f"brier_{c}"}),
                        on=["event_ticker", "horizon"])
    return out


def _event_table(ev: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for (part, hz), g in ev.groupby([label_col, "horizon"]):
        g = g.assign(d_ls=g["ls_p_blend"] - g["ls_p_mkt"])
        lo, hi = _date_boot(g, "d_ls")
        rows.append({"part": part, "horizon": hz, "events": len(g), "dates": g["target_date"].nunique(),
                     "logscore_market": g["ls_p_mkt"].mean(), "logscore_model": g["ls_p_model"].mean(),
                     "logscore_blend": g["ls_p_blend"].mean(), "blend_minus_market": g["d_ls"].mean(),
                     "ci90_lo": lo, "ci90_hi": hi,
                     "brier_market": g["brier_p_mkt"].mean(), "brier_blend": g["brier_p_blend"].mean()})
    return pd.DataFrame(rows)


def _strategy(con, col, label, pick=None, pick_label="DEV"):
    """Sweep thresholds on the `pick` rows (default: all DEV), report VALIDATION at the choice."""
    pick = (con["split"] == "dev") if pick is None else pick
    out, rows, chosen = [], [], {}
    for hz in ("open", "eve"):
        dev = con[pick & (con["horizon"] == hz)]
        best = None
        for th in THRESHOLDS:
            r = dict(horizon=hz, threshold=th, **_summ(trades(dev, th, col)))
            rows.append(r)
            if r["trades"] >= 100 and (best is None or r["pnl_per_contract"] > best[1]):
                best = (th, r["pnl_per_contract"])
        chosen[hz] = best[0] if best else None
    out += [f"[{label}] {pick_label} threshold sweep (choose here, never on validation):",
            _fmt(pd.DataFrame(rows)), ""]
    rows = []
    for hz, th in chosen.items():
        if th is None:
            continue
        hold = con[(con["split"] == "validation") & (con["horizon"] == hz)]
        rows.append(dict(horizon=hz, threshold=th, **_summ(trades(hold, th, col))))
    out += [f"[{label}] VALIDATION at the {pick_label}-chosen threshold:",
            _fmt(pd.DataFrame(rows)) if rows else f"(not enough {pick_label} trades to choose a threshold)", ""]
    allt = [trades(con[con["horizon"] == hz], th, col) for hz, th in chosen.items() if th is not None]
    allt = pd.concat(allt) if allt else pd.DataFrame()
    if not allt.empty:
        allt["edge_bin"] = pd.cut(allt["exp_edge"], [0, .03, .06, .1, .2, 1.0])
        eb = allt.groupby(["split", "edge_bin"], observed=True).agg(
            trades=("pnl", "size"), exp_edge=("exp_edge", "mean"), realized=("pnl", "mean")).reset_index()
        out += [f"[{label}] Promised vs delivered edge (realized << exp_edge means overconfident):", _fmt(eb), ""]
        allt["price_bin"] = pd.cut(allt["price"], [0, .2, .4, .6, .8, 1.0])
        pbn = allt.groupby(["split", "price_bin"], observed=True).agg(
            trades=("pnl", "size"), realized=("pnl", "mean")).reset_index()
        out += [f"[{label}] Realized PnL per contract by price paid:", _fmt(pbn), ""]
    return out, chosen


def phase3(con: pd.DataFrame) -> str:
    out =["PHASE 3 - paper backtest. Buy YES at the ask or NO at (1 - bid), taker fees, "
           f"{CONTRACTS_PER_ORDER}-contract orders, 1 decision per contract per horizon.",
           "Fills assume the quoted price was available for the size; real fills will be worse.", ""]
    con, coefs = add_blend(con)
    q = quoted(con)
    out += [f"Quoted contracts: dev fit={int((q['dev_part'] == 'fit').sum())}  "
            f"dev tune={int((q['dev_part'] == 'tune').sum())}  "
            f"validation={int((q['split'] == 'validation').sum())}",
            "P&L confidence intervals resample whole calendar dates.", ""]

    lines, _ = _strategy(con, "p_model", "v1 model alone")
    out += lines
    if not coefs:
        out += [f"[v2 market+model blend] pending: needs >= {MIN_BLEND_FIT} quotes per horizon "
                "on the earlier DEV dates.", ""]
        return "\n".join(out)

    ct = pd.DataFrame([{"horizon": h, "b0": c["b0"], "b_market": c["b_market"],
                        "b_market_lo": c["b_market_ci90"][0], "b_market_hi": c["b_market_ci90"][1],
                        "b_model": c["b_model"], "b_model_lo": c["b_model_ci90"][0],
                        "b_model_hi": c["b_model_ci90"][1], "contracts": c["n_contracts"],
                        "events": c["n_events"], "dates": c["n_dates"]} for h, c in coefs.items()])
    out += ["[v2 market+model blend] coefficients fit on the earlier DEV dates (logit scale),",
            "90% CI from resampling dates. b_model's CI excluding 0 = the model adds information:",
            _fmt(ct), ""]
    rows = []
    for (part, hz), g in q[q["p_blend"].notna() & (q["dev_part"] != "fit")].assign(
            part=lambda d: d["dev_part"].fillna(d["split"])).groupby(["part", "horizon"]):
        y = g["y"].to_numpy()
        rows.append({"part": part, "horizon": hz, "contracts": len(g), "events": g["event_ticker"].nunique(),
                     "dates": g["target_date"].nunique(), "brier_market": _brier(g["p_mkt"], y),
                     "brier_blend": _brier(g["p_blend"], y), "logloss_market": _logloss(g["p_mkt"], y),
                     "logloss_blend": _logloss(g["p_blend"], y)})
    out += ["[v2] Per contract, out of sample for the blend weights (tune = later DEV dates):",
            _fmt(pd.DataFrame(rows)), ""]
    ev = event_scores(con[con["dev_part"] != "fit"], ["p_mkt", "p_model", "p_blend"])
    if not ev.empty:
        ev["part"] = ev["dev_part"].fillna(ev["split"])
        out += ["[v2] Per event (all brackets quoted, renormalised to sum to 1). blend_minus_market < 0",
                "with ci90_hi < 0 = the blend beats the market price; CI resamples whole dates:",
                _fmt(_event_table(ev, "part")), ""]
    lines, chosen = _strategy(con, "p_blend", "v2 market+model blend",
                              pick=con["dev_part"] == "tune", pick_label="DEV tune")
    out += lines
    out += [_freeze_blend(coefs, chosen), ""]
    return "\n".join(out)


def run(conn, params=None, reuse=False) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    saved = sorted(REPORT_DIR.glob("predictions_*.csv"))
    if reuse and saved:   # skip the slow refit; contract probabilities/quotes are still recomputed
        preds = pd.read_csv(saved[-1])
        preds["decision"] = pd.to_datetime(preds["decision"], utc=True)
    else:
        preds = predictions(conn, params)
    if preds.empty:
        return "No predictions: run `backfill-history` first."
    con = contracts(conn, preds)
    con["split"] = split_of(con["target_date"])
    stamp = date.today().isoformat()
    preds.to_csv(REPORT_DIR / f"predictions_{stamp}.csv", index=False)
    con.drop(columns=["decision"]).to_csv(REPORT_DIR / f"contracts_{stamp}.csv", index=False)
    header = (f"Isotherm evaluation  model={model.VERSION}  params={dict(model.DEFAULTS, **(params or {}))}\n"
              f"prediction rows={len(preds)}  contracts={len(con)}  "
              f"with quotes={int(con['yes_ask'].notna().sum())}  validation from {HOLDOUT_START} "
              "(inspected during development; forward paper trading is the clean test)\n")
    text = header + "\n" + phase2(preds, con) + "\n" + phase3(con)
    (REPORT_DIR / f"report_{stamp}.txt").write_text(text, encoding="utf-8")
    return text
