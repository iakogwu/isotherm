"""Phase 2: calibrated forecast distribution for a station's daily high/low.

Model (per station x kind x horizon, refit every day on a trailing window):

    base  = NBS max/min if available, else mean of the NWP models
    mean  = base + b0 + sum_i b_i * (model_i - base)        ridge regression
    var   = a + b * nbs_sd^2 + c * model_spread^2              fit by max likelihood
                                                              on leave-one-out residuals
    P(official value = n) = Normal(mean, sd) mass on [n - 0.5, n + 0.5)

Training rows for target day D only use outcomes up to D-2, which were published before
either decision time. Everything the model sees is leak-free (see features.py).
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

from . import features

VERSION = "emos-ridge-v2"
DEFAULTS = {"window": 60, "min_train": 30, "ridge": 10.0, "sigma_floor": 0.7}
# Bracket probabilities use a Student-t with these settings. Chosen on DEV contracts only
# (2026-10-04): Gaussian tails were too thin (1.4% predicted vs 2.6% observed).
TAIL_DF = 5
SIGMA_SCALE = 0.9
MODEL_COLS = list(features.HIST_MODELS)


# ------------------------------------------------------------------ dataset

def _nbs_lookup(conn, rows: pd.DataFrame, station_id=None) -> pd.DataFrame:
    """Attach the latest NBS max/min (and its sd) that was public at each decision time."""
    q = ("SELECT station_id, runtime_utc, ftime_utc, value AS nbs, spread AS nbs_sd "
         "FROM mos_forecasts WHERE model='NBS'")
    mos = pd.read_sql_query(q + (" AND station_id=?" if station_id else ""), conn,
                            params=(station_id,) if station_id else None)
    if mos.empty:
        rows["nbs"] = np.nan
        rows["nbs_sd"] = np.nan
        return rows
    mos["runtime"] = pd.to_datetime(mos["runtime_utc"].str[:16], utc=True)
    mos = mos[mos["runtime"].dt.hour.isin(features.NBS_RUN_HOURS)]
    mos["ftime"] = mos["ftime_utc"].str[:16]
    mos = mos.sort_values("runtime")
    rows = rows.copy()
    rows["cutoff"] = rows["decision"] - features.NBS_LATENCY
    rows = rows.sort_values("cutoff")
    out = pd.merge_asof(rows, mos[["station_id", "ftime", "runtime", "nbs", "nbs_sd"]],
                        left_on="cutoff", right_on="runtime", by=["station_id", "ftime"],
                        direction="backward")
    return out.drop(columns=["cutoff"])


def build_dataset(conn, horizon: str, station_id=None) -> pd.DataFrame:
    hf = pd.read_sql_query(
        "SELECT station_id, target_date, kind, model, value FROM hist_forecasts WHERE horizon=?"
        + (" AND station_id=?" if station_id else ""),
        conn, params=(horizon, station_id) if station_id else (horizon,))
    if hf.empty:
        return hf
    wide = hf.pivot_table(index=["station_id", "target_date", "kind"], columns="model",
                          values="value").reset_index()
    for m in MODEL_COLS:
        if m not in wide:
            wide[m] = np.nan
    st = pd.read_sql_query("SELECT station_id, tz FROM stations", conn)
    wide = wide.merge(st, on="station_id")
    dates = pd.to_datetime(wide["target_date"]).dt.date
    wide["decision"] = pd.to_datetime(
        [features.decision_time(tz, d, horizon) for tz, d in zip(wide["tz"], dates)], utc=True)
    wide["ftime"] = [features.nbs_ftime(d, k) for d, k in zip(dates, wide["kind"])]
    wide = _nbs_lookup(conn, wide, station_id)

    # observed: Kalshi official value first, NWS climate report as fallback
    oc = pd.read_sql_query("SELECT station_id, target_date, kind, value AS obs_k FROM outcomes", conn)
    oc = oc.groupby(["station_id", "target_date", "kind"], as_index=False)["obs_k"].first()
    cli = pd.read_sql_query("SELECT station_id, date AS target_date, high, low FROM cli_reports", conn)
    cli = cli.melt(id_vars=["station_id", "target_date"], value_vars=["high", "low"],
                   var_name="kind", value_name="obs_c")
    wide = wide.merge(oc, on=["station_id", "target_date", "kind"], how="left")
    wide = wide.merge(cli, on=["station_id", "target_date", "kind"], how="left")
    wide["obs"] = wide["obs_k"].fillna(wide["obs_c"])
    wide["horizon"] = horizon
    return wide.drop(columns=["obs_k", "obs_c"]).sort_values(
        ["station_id", "kind", "target_date"]).reset_index(drop=True)


# ------------------------------------------------------------------ fitting

def _design(df: pd.DataFrame):
    X_models = df[MODEL_COLS].to_numpy(dtype=float)
    any_model = ~np.isnan(X_models).all(axis=1)
    model_mean = np.full(len(df), np.nan)
    spread = np.full(len(df), np.nan)
    model_mean[any_model] = np.nanmean(X_models[any_model], axis=1)
    spread[any_model] = np.nanstd(X_models[any_model], axis=1)
    nbs = df["nbs"].to_numpy(dtype=float)
    base = np.where(np.isnan(nbs), model_mean, nbs)
    # a missing model contributes "no opinion" (diff 0) rather than NaN
    diffs = np.nan_to_num(X_models - base[:, None], nan=0.0)
    X = np.column_stack([np.ones(len(df)), diffs])
    nbs_sd = df["nbs_sd"].to_numpy(dtype=float)
    return X, base, spread, nbs_sd


def _fit_var(resid, nbs_sd, spread, floor):
    sd_fill = np.where(np.isnan(nbs_sd), np.nanmedian(nbs_sd) if np.any(~np.isnan(nbs_sd)) else 2.0, nbs_sd)
    sp = np.nan_to_num(spread)

    def nll(p):
        v = p[0] + p[1] * sd_fill ** 2 + p[2] * sp ** 2
        v = np.maximum(v, floor ** 2)
        return 0.5 * np.sum(np.log(v) + resid ** 2 / v)

    x0 = np.array([max(np.var(resid), floor ** 2), 0.0, 0.0])
    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(1e-3, None), (0, 5), (0, 5)])
    return res.x, np.nanmedian(sd_fill)


def fit_predict(df: pd.DataFrame, params=None, only_dates=None) -> pd.DataFrame:
    """Rolling out-of-sample predictions for every row with enough history
    (or just `only_dates`, an iterable of ISO dates, for live use)."""
    only = set(only_dates) if only_dates else None
    p = dict(DEFAULTS, **(params or {}))
    out = []
    for (sid, kind), g in df.groupby(["station_id", "kind"], sort=False):
        g = g.reset_index(drop=True)
        g = g[~(g[MODEL_COLS].isna().all(axis=1) & g["nbs"].isna())].reset_index(drop=True)
        if g.empty:
            continue
        X, base, spread, nbs_sd = _design(g)
        y = g["obs"].to_numpy(dtype=float)
        dates = pd.to_datetime(g["target_date"]).to_numpy()
        lam = np.diag([0.0] + [p["ridge"]] * (X.shape[1] - 1))
        for i in range(len(g)):
            if only is not None and g.at[i, "target_date"] not in only:
                continue
            hi = dates[i] - np.timedelta64(2, "D")
            lo = hi - np.timedelta64(p["window"], "D")
            tr = (dates <= hi) & (dates > lo) & ~np.isnan(y)
            n = int(tr.sum())
            if n < p["min_train"]:
                continue
            Xt, rt = X[tr], (y - base)[tr]
            A = np.linalg.inv(Xt.T @ Xt + lam)
            beta = A @ Xt.T @ rt
            h = np.einsum("ij,jk,ik->i", Xt, A, Xt)
            loo = (rt - Xt @ beta) / np.clip(1 - h, 0.05, None)
            (a, b, c), sd_med = _fit_var(loo, nbs_sd[tr], spread[tr], p["sigma_floor"])
            sdi = nbs_sd[i] if not np.isnan(nbs_sd[i]) else sd_med
            var = a + b * sdi ** 2 + c * (0 if np.isnan(spread[i]) else spread[i]) ** 2
            out.append({
                "station_id": sid, "kind": kind, "target_date": g.at[i, "target_date"],
                "horizon": g.at[i, "horizon"], "decision": g.at[i, "decision"],
                "mu": float(base[i] + X[i] @ beta),
                "sigma": float(max(math.sqrt(max(var, 0)), p["sigma_floor"])),
                "nbs": g.at[i, "nbs"], "nbs_sd": g.at[i, "nbs_sd"], "obs": y[i], "n_train": n,
            })
    return pd.DataFrame(out)


# ------------------------------------------------------------------ probabilities

def bracket_bounds(strike_type, floor, cap):
    """Integer range [lo, hi] that makes a contract resolve YES."""
    if strike_type == "greater":
        return floor + 1, math.inf
    if strike_type == "less":
        return -math.inf, cap - 1
    if strike_type == "between":
        return floor, cap
    return None


def prob_in(mu, sigma, lo, hi):
    """P(rounded value in [lo, hi]) under a scaled Student-t around mu. Vectorised."""
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float) * SIGMA_SCALE
    cdf = (lambda z: student_t.cdf(z, TAIL_DF)) if TAIL_DF else norm.cdf
    upper = np.where(np.isinf(hi), 1.0, cdf((np.asarray(hi, float) + 0.5 - mu) / sigma))
    lower = np.where(np.isinf(lo), 0.0, cdf((np.asarray(lo, float) - 0.5 - mu) / sigma))
    return np.clip(upper - lower, 0.0, 1.0)


def crps_gaussian(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def crps_traded(mu, sigma, y):
    """CRPS of the distribution bracket probabilities actually use: Student-t(TAIL_DF) at
    sigma * SIGMA_SCALE (closed form, Jordan et al. 2019, scoringRules)."""
    from scipy.special import beta
    s = np.asarray(sigma, float) * SIGMA_SCALE
    if not TAIL_DF:
        return crps_gaussian(mu, s, y)
    v = TAIL_DF
    z = (np.asarray(y, float) - np.asarray(mu, float)) / s
    return s * (z * (2 * student_t.cdf(z, v) - 1)
                + 2 * student_t.pdf(z, v) * (v + z ** 2) / (v - 1)
                - 2 * math.sqrt(v) * beta(0.5, v - 0.5) / ((v - 1) * beta(0.5, v / 2) ** 2))


def central_z(level=0.8):
    """Half-width, in units of sigma, of the traded distribution's central `level` interval."""
    q = (1 + level) / 2
    return SIGMA_SCALE * (student_t.ppf(q, TAIL_DF) if TAIL_DF else norm.ppf(q))
