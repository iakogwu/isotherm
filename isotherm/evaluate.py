"""Phase 2 (forecast skill) and Phase 3 (paper-trading backtest) reports.

    python -m isotherm evaluate

Split discipline:
  DEV      target dates before HOLDOUT_START: free to look at, tune, iterate.
  HOLDOUT  HOLDOUT_START onward: reported only at settings chosen on DEV.
  FORWARD  paper_predictions logged live by the scheduler. The final word.
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
    return np.where(pd.Series(d).astype(str) < HOLDOUT_START, "dev", "holdout")


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
    p["crps_model"] = model.crps_gaussian(p["mu"], p["sigma"], p["obs"])
    has = p["nbs"].notna()
    p["crps_nbs"] = np.nan
    p.loc[has, "crps_nbs"] = model.crps_gaussian(
        p.loc[has, "nbs"], np.maximum(p.loc[has, "nbs_sd"].fillna(2.0), 0.7), p.loc[has, "obs"])
    p["ae_model"] = (p["mu"] - p["obs"]).abs()
    p["ae_nbs"] = (p["nbs"] - p["obs"]).abs()
    z = (p["obs"] - p["mu"]) / p["sigma"]
    p["in80"] = (z.abs() <= 1.2816).astype(float)
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

    st = pb[pb["split"] == "holdout"].groupby(["station_id", "kind"]).agg(
        days=("obs", "size"), crps_model=("crps_model", "mean"), crps_nbs=("crps_nbs", "mean")).reset_index()
    st["skill"] = 1 - st["crps_model"] / st["crps_nbs"]
    out += ["HOLDOUT skill vs NBS by station (positive = model better):",
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


def _boot_ci(t: pd.DataFrame, n_boot=2000, seed=7):
    """90% CI for mean PnL/contract, resampling whole (station, day) clusters - bets on the
    same day and city are correlated, so contract-level resampling would overstate certainty."""
    if t.empty:
        return (np.nan, np.nan)
    g = t.groupby(["station_id", "target_date"])["pnl"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return tuple(np.percentile(means, [5, 95]))


def _summ(t):
    lo, hi = _boot_ci(t)
    return {"trades": len(t), "win_rate": t["won"].mean() if len(t) else np.nan,
            "avg_price": t["price"].mean() if len(t) else np.nan,
            "exp_edge": t["exp_edge"].mean() if len(t) else np.nan,
            "pnl_per_contract": t["pnl"].mean() if len(t) else np.nan,
            "ci90_lo": lo, "ci90_hi": hi, "total_pnl_$1": t["pnl"].sum()}


BLEND_FILE = REPORT_DIR / "blend.json"
MIN_BLEND_FIT = 2000


def add_blend(con: pd.DataFrame):
    """Fit the market+model blend per horizon on DEV quotes only; add p_blend to quoted rows."""
    q = quoted(con)
    con = con.copy()
    con["p_mkt"] = np.nan
    con.loc[q.index, "p_mkt"] = q["p_mkt"]
    con["p_blend"] = np.nan
    coefs = {}
    for hz in ("open", "eve"):
        dev = q[(q["split"] == "dev") & (q["horizon"] == hz)]
        if len(dev) < MIN_BLEND_FIT:
            continue
        b = fit_blend(dev)
        coefs[hz] = {"b0": b[0], "b_market": b[1], "b_model": b[2], "n_dev": len(dev)}
        rows = q[q["horizon"] == hz]
        con.loc[rows.index, "p_blend"] = apply_blend(rows, b)
    return con, coefs


def _strategy(con, col, label):
    out, rows, chosen = [], [], {}
    for hz in ("open", "eve"):
        dev = con[(con["split"] == "dev") & (con["horizon"] == hz)]
        best = None
        for th in THRESHOLDS:
            r = dict(horizon=hz, threshold=th, **_summ(trades(dev, th, col)))
            rows.append(r)
            if r["trades"] >= 100 and (best is None or r["pnl_per_contract"] > best[1]):
                best = (th, r["pnl_per_contract"])
        chosen[hz] = best[0] if best else None
    out += [f"[{label}] DEV threshold sweep (choose here, never on holdout):", _fmt(pd.DataFrame(rows)), ""]
    rows = []
    for hz, th in chosen.items():
        if th is None:
            continue
        hold = con[(con["split"] == "holdout") & (con["horizon"] == hz)]
        rows.append(dict(horizon=hz, threshold=th, **_summ(trades(hold, th, col))))
    out += [f"[{label}] HOLDOUT at the DEV-chosen threshold (the honest number):",
            _fmt(pd.DataFrame(rows)) if rows else "(not enough DEV trades to choose a threshold)", ""]
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
    import json
    out = ["PHASE 3 - paper backtest. Buy YES at the ask or NO at (1 - bid), taker fees, "
           f"{CONTRACTS_PER_ORDER}-contract orders, 1 decision per contract per horizon.",
           "Fills assume the quoted price was available for the size; real fills will be worse.", ""]
    con, coefs = add_blend(con)
    q = quoted(con)
    out += [f"Quoted contracts: dev={int((q['split'] == 'dev').sum())}  holdout={int((q['split'] == 'holdout').sum())}", ""]

    lines, _ = _strategy(con, "p_model", "v1 model alone")
    out += lines
    if not coefs:
        out += [f"[v2 market+model blend] pending: needs >= {MIN_BLEND_FIT} DEV quotes per horizon "
                "(older Kalshi price history still downloading).", ""]
        return "\n".join(out)

    out += ["[v2 market+model blend] coefficients fit on DEV (logit scale):",
            _fmt(pd.DataFrame([dict(horizon=h, **c) for h, c in coefs.items()])), ""]
    hb = q[(q["split"] == "holdout") & q["p_blend"].notna()]
    rows = []
    for hz, g in hb.groupby("horizon"):
        y = g["y"].to_numpy()
        rows.append({"horizon": hz, "contracts": len(g), "brier_market": _brier(g["p_mkt"], y),
                     "brier_blend": _brier(g["p_blend"], y), "logloss_market": _logloss(g["p_mkt"], y),
                     "logloss_blend": _logloss(g["p_blend"], y)})
    out += ["[v2] HOLDOUT accuracy: does the blend beat the market price itself?", _fmt(pd.DataFrame(rows)), ""]
    lines, chosen = _strategy(con, "p_blend", "v2 market+model blend")
    out += lines
    BLEND_FILE.write_text(json.dumps({"model_version": model.VERSION, "fit_on": f"dev < {HOLDOUT_START}",
                                      "coefs": coefs, "thresholds": chosen}, indent=2))
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
              f"with quotes={int(con['yes_ask'].notna().sum())}  holdout from {HOLDOUT_START}\n")
    text = header + "\n" + phase2(preds, con) + "\n" + phase3(con)
    (REPORT_DIR / f"report_{stamp}.txt").write_text(text, encoding="utf-8")
    return text
