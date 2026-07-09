"""
Module 7 - Backtest / Validation Engine.

Runs on demand, NOT on the 5-minute cron - this pulls months of real
historical data and replays the EXACT SAME functions from bot.py (Modules
3-5: feature engine, scoring engine, risk engine) bar by bar. That reuse is
the entire point: if this used separate reimplemented logic, a good result
here wouldn't actually say anything about the live system's edge.

What this validates: whether the CURRENT fixed thresholds/weights (ADX
floor 22, publish floor 78, the 6 scoring weights, R:R minimum 1.5, etc.)
would have produced a profitable, honest signal stream against real price
history. What this does NOT do yet: walk-forward re-splitting, Monte Carlo
resampling, or parameter sensitivity sweeps - those are follow-up passes
once this baseline result has actually been reviewed. Reporting a single
backtest as "validated" without those would overstate what one pass proves.
"""

import time

import numpy as np
import pandas as pd
import requests

from bot import (
    TWELVEDATA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TWELVEDATA_SYMBOL, INSTRUMENT,
    ATR_PERIOD, ATR_LOOKBACK, ADX_PERIOD, EMA_FAST, EMA_SLOW, EMA_SLOPE_LOOKBACK,
    STRUCTURE_LOOKBACK, ROC_LOOKBACK, PUBLISH_FLOOR,
    compute_atr, compute_ema, compute_adx, compute_session_vwap, get_session,
    determine_candidate_direction, check_regime_gate, compute_composite_score,
    compute_risk_levels, send_telegram_message,
)

# Targets, not guarantees - fetch_historical_paginated stops early and
# gracefully if the feed runs out of history before reaching these.
BACKTEST_M5_TARGET_BARS = 100_000   # ~ a year of M5 bars, if available
BACKTEST_H1_TARGET_BARS = 9_000     # ~ a year of H1 bars, if available
PACE_SECONDS = 8.0                  # keeps requests under the free tier's 8 credits/minute
MAX_TRADE_BARS_FORWARD = 500        # ~ 41h of M5 bars to let a trade resolve before calling it "still open"


# ---------------------------------------------------------------------------
# Diagnostic categorization - what we can HONESTLY measure with the current
# feature set. This is NOT the full category list an institutional review
# would want (news events, a distinct mean-reversion regime, late-entry vs
# early-exit timing all require detectors this system doesn't have yet) -
# it's exactly what's measurable today, named accurately rather than padded
# out with categories we'd have to fake.
# ---------------------------------------------------------------------------

def _adx_bucket(adx: float) -> str:
    if adx < 25: return "22-25 (barely trending)"
    if adx < 30: return "25-30"
    if adx < 40: return "30-40"
    return "40+ (strong trend)"


def _atr_pct_bucket(pct: float) -> str:
    if pct < 50: return "30-50 (low-normal)"
    if pct < 80: return "50-80 (constructive)"
    return "80-95 (high/spike zone)"


def _confidence_bucket(conf: float) -> str:
    if conf < 82: return "78-82"
    if conf < 90: return "82-90"
    return "90-100"


def _structure_basis(structure: dict, direction: str) -> str:
    agreeing = structure["bullish_reclaim"] if direction == "BUY" else structure["bearish_reclaim"]
    opposing = structure["bearish_reclaim"] if direction == "BUY" else structure["bullish_reclaim"]
    if agreeing:
        return "genuine reclaim"
    if opposing:
        return "opposing reclaim (should not occur post-gate)"
    if structure["swept_high"] or structure["swept_low"]:
        return "swept, no reclaim"
    return "no structure event"


# ---------------------------------------------------------------------------
# Historical data fetch (paginated - single calls cap at 5000 bars)
# ---------------------------------------------------------------------------

def fetch_historical_paginated(symbol: str, interval: str, target_bars: int, api_key: str,
                                pace_seconds: float = PACE_SECONDS, chunk_size: int = 5000) -> pd.DataFrame:
    """
    Twelve Data caps outputsize at 5000/call. Walks backward using end_date
    anchoring (a documented pagination pattern) until target_bars is
    covered or the feed runs out of history - whichever comes first.
    """
    url = "https://api.twelvedata.com/time_series"
    chunks = []
    end_date = None
    collected = 0

    while collected < target_bars:
        params = {"symbol": symbol, "interval": interval, "outputsize": chunk_size,
                  "timezone": "UTC", "apikey": api_key}
        if end_date is not None:
            params["end_date"] = end_date.strftime("%Y-%m-%d %H:%M:%S")

        data = None
        last_error = None
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=30)
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                time.sleep(pace_seconds)
                data = None
                continue

            if isinstance(data, dict) and data.get("status") == "error":
                last_error = data.get("message", "unknown error")
                rate_limited = resp.status_code == 429 or "limit" in str(last_error).lower()
                if rate_limited:
                    time.sleep(pace_seconds * 2)
                    data = None
                    continue
                raise RuntimeError(f"Twelve Data API error fetching {symbol} {interval}: {last_error}")
            break

        if data is None:
            raise RuntimeError(f"Twelve Data request failed repeatedly ({symbol} {interval}): {last_error}")

        values = data.get("values")
        if not values:
            print(f"No further history for {symbol} {interval} before {end_date} - stopping pagination")
            break

        df_chunk = pd.DataFrame(values)
        df_chunk["time"] = pd.to_datetime(df_chunk["datetime"], utc=True)
        for col in ("open", "high", "low", "close"):
            df_chunk[col] = df_chunk[col].astype(float)
        df_chunk = df_chunk.sort_values("time").reset_index(drop=True)

        chunks.append(df_chunk[["time", "open", "high", "low", "close"]])
        collected += len(df_chunk)
        end_date = df_chunk["time"].iloc[0] - pd.Timedelta(seconds=1)

        if len(df_chunk) < chunk_size:
            break  # short chunk = hit the start of available history

        time.sleep(pace_seconds)

    if not chunks:
        raise RuntimeError(f"No historical data retrieved at all for {symbol} {interval}")

    full = pd.concat(chunks, ignore_index=True)
    return full.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Vectorized feature computation (O(n), not O(n^2))
# ---------------------------------------------------------------------------

def compute_structure_series(df: pd.DataFrame, lookback: int = STRUCTURE_LOOKBACK) -> pd.DataFrame:
    """
    Vectorized equivalent of bot.detect_liquidity_sweep() for every bar at
    once. Verified in tests to produce identical results to calling
    detect_liquidity_sweep() at each corresponding index - this is what
    lets the backtest loop be O(n) instead of recomputing from scratch
    per bar.
    """
    prior_high = df["high"].shift(1).rolling(lookback).max()
    prior_low = df["low"].shift(1).rolling(lookback).min()
    swept_high = df["high"] > prior_high
    swept_low = df["low"] < prior_low
    bullish_reclaim = swept_low & (df["close"] > prior_low)
    bearish_reclaim = swept_high & (df["close"] < prior_high)
    return pd.DataFrame({
        "prior_high": prior_high, "prior_low": prior_low,
        "swept_high": swept_high, "swept_low": swept_low,
        "bullish_reclaim": bullish_reclaim, "bearish_reclaim": bearish_reclaim,
    })


def _pct_rank_of_last(arr: np.ndarray) -> float:
    last = arr[-1]
    return (arr < last).sum() / len(arr) * 100


def build_h1_feature_frame(h1_df: pd.DataFrame) -> pd.DataFrame:
    """
    H1-derived features, each stamped with when it actually becomes
    available (1 hour after the bar's own start - when it's actually
    closed). Used to join onto M5 timestamps without lookahead: an M5 bar
    at 10:03 must never see the H1 bar that opened at 10:00, since that H1
    bar hasn't closed yet in real time.
    """
    adx_series = compute_adx(h1_df, period=ADX_PERIOD)
    ema50 = compute_ema(h1_df["close"], EMA_SLOW)
    bias = np.where(h1_df["close"] > ema50, "Bullish", "Bearish")

    out = pd.DataFrame({
        "h1_time": h1_df["time"],
        "adx_h1": adx_series,
        "htf_bias": bias,
    })
    out["available_at"] = out["h1_time"] + pd.Timedelta(hours=1)
    return out


def align_h1_to_m5(m5_df: pd.DataFrame, h1_features: pd.DataFrame) -> pd.DataFrame:
    """merge_asof(direction='backward') guarantees only already-closed H1 bars are ever used."""
    left = m5_df[["time"]].sort_values("time")
    right = h1_features.sort_values("available_at")
    merged = pd.merge_asof(left, right, left_on="time", right_on="available_at", direction="backward")
    return merged[["adx_h1", "htf_bias"]].reset_index(drop=True)


def build_feature_frame(m5_df: pd.DataFrame, h1_df: pd.DataFrame) -> pd.DataFrame:
    """
    One pass, full history, every feature Module 3 computes - as a frame
    instead of a single latest-value snapshot. Each row is exactly what
    compute_features() would have returned if called live at that bar's
    close (verified in tests by spot-checking against compute_features()
    called directly on truncated history).
    """
    atr = compute_atr(m5_df, period=ATR_PERIOD)
    ema_fast = compute_ema(m5_df["close"], EMA_FAST)
    ema_slow = compute_ema(m5_df["close"], EMA_SLOW)
    roc = m5_df["close"].pct_change(ROC_LOOKBACK) * 100
    slope = (ema_fast - ema_fast.shift(EMA_SLOPE_LOOKBACK)) / ema_fast.shift(EMA_SLOPE_LOOKBACK) * 100
    structure = compute_structure_series(m5_df, lookback=STRUCTURE_LOOKBACK)
    atr_pct = atr.rolling(ATR_LOOKBACK, min_periods=max(20, ATR_LOOKBACK // 3)).apply(_pct_rank_of_last, raw=True)
    session = m5_df["time"].dt.hour.apply(get_session)

    h1_features = build_h1_feature_frame(h1_df)
    h1_aligned = align_h1_to_m5(m5_df, h1_features)

    frame = pd.DataFrame({
        "time": m5_df["time"].values,
        "close": m5_df["close"].values,
        "high": m5_df["high"].values,
        "low": m5_df["low"].values,
        "atr_m5": atr.values,
        "atr_percentile": atr_pct.values,
        "trend_direction": np.where(ema_fast > ema_slow, "Bullish", "Bearish"),
        "trend_slope_pct": slope.values,
        "roc_pct": roc.values,
        "session": session.values,
    })
    frame = pd.concat([frame, structure.reset_index(drop=True)], axis=1)
    frame["adx_h1"] = h1_aligned["adx_h1"].values
    frame["htf_bias"] = h1_aligned["htf_bias"].values
    return frame


# ---------------------------------------------------------------------------
# Backtest loop - reuses bot.py's live scoring/risk functions unchanged
# ---------------------------------------------------------------------------

def simulate_trade(frame: pd.DataFrame, entry_index: int, direction: str, levels: dict,
                    max_bars_forward: int = MAX_TRADE_BARS_FORWARD) -> dict:
    """
    Walks forward using REAL subsequent bars to see whether TP or SL was
    touched first. Both touched in the same bar -> counted as a loss:
    without tick data there's no way to know which was actually hit first
    intra-bar, and assuming the favorable outcome would bias results optimistic.
    """
    sl, tp = levels["stop_loss"], levels["take_profit"]
    n = len(frame)
    end = min(entry_index + 1 + max_bars_forward, n)

    for j in range(entry_index + 1, end):
        bar = frame.iloc[j]
        hit_tp = bar["high"] >= tp if direction == "BUY" else bar["low"] <= tp
        hit_sl = bar["low"] <= sl if direction == "BUY" else bar["high"] >= sl

        if hit_tp and hit_sl:
            return {"outcome": "loss", "r_multiple": -1.0, "bars_to_resolve": j - entry_index,
                    "resolution": "ambiguous_same_bar_conservative_loss"}
        if hit_tp:
            r = levels["tp_distance"] / levels["sl_distance"]
            return {"outcome": "win", "r_multiple": r, "bars_to_resolve": j - entry_index, "resolution": "tp_hit"}
        if hit_sl:
            return {"outcome": "loss", "r_multiple": -1.0, "bars_to_resolve": j - entry_index, "resolution": "sl_hit"}

    return {"outcome": "open", "r_multiple": 0.0, "bars_to_resolve": None, "resolution": "unresolved_within_window"}


def run_backtest(frame: pd.DataFrame, warmup_bars: int) -> list:
    results = []
    n = len(frame)

    for i in range(warmup_bars, n):
        row = frame.iloc[i]
        if pd.isna(row["adx_h1"]) or pd.isna(row["atr_percentile"]) or pd.isna(row["trend_slope_pct"]) or pd.isna(row["roc_pct"]):
            continue

        structure = {
            "prior_high": row["prior_high"], "prior_low": row["prior_low"],
            "swept_high": bool(row["swept_high"]), "swept_low": bool(row["swept_low"]),
            "bullish_reclaim": bool(row["bullish_reclaim"]), "bearish_reclaim": bool(row["bearish_reclaim"]),
        }
        features = {
            "adx_h1": row["adx_h1"], "atr_percentile": row["atr_percentile"],
            "trend_direction": row["trend_direction"], "htf_bias": row["htf_bias"],
            "trend_slope_pct": row["trend_slope_pct"], "roc_pct": row["roc_pct"],
            "session": row["session"], "structure": structure, "atr_m5": row["atr_m5"],
        }

        direction = determine_candidate_direction(features)
        if direction is None:
            continue
        gate_ok, _ = check_regime_gate(features)
        if not gate_ok:
            continue

        scoring = compute_composite_score(features, direction)
        if scoring["composite"] < PUBLISH_FLOOR:
            continue

        levels = compute_risk_levels(row["close"], row["atr_m5"], direction, structure)
        outcome = simulate_trade(frame, i, direction, levels)

        results.append({
            "time": row["time"], "direction": direction, "confidence": round(scoring["composite"], 1),
            "entry": levels["entry"], "stop_loss": levels["stop_loss"], "take_profit": levels["take_profit"],
            "session": row["session"],
            "adx_bucket": _adx_bucket(row["adx_h1"]),
            "atr_pct_bucket": _atr_pct_bucket(row["atr_percentile"]),
            "confidence_bucket": _confidence_bucket(scoring["composite"]),
            "structure_basis": _structure_basis(structure, direction),
            "stop_basis": levels["stop_basis"],
            "tp_basis": levels["tp_basis"],
            **outcome,
        })

    return results


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list) -> dict:
    resolved = [r for r in results if r["outcome"] in ("win", "loss")]
    wins = [r for r in resolved if r["outcome"] == "win"]
    losses = [r for r in resolved if r["outcome"] == "loss"]
    open_trades = [r for r in results if r["outcome"] == "open"]

    total = len(resolved)
    win_rate = (len(wins) / total * 100) if total else 0.0
    avg_r = (sum(r["r_multiple"] for r in resolved) / total) if total else 0.0
    total_r = sum(r["r_multiple"] for r in resolved)

    r_values = [r["r_multiple"] for r in resolved]
    equity_curve = np.cumsum(r_values) if r_values else np.array([0.0])
    running_max = np.maximum.accumulate(equity_curve)
    max_drawdown_r = float((running_max - equity_curve).max())

    gross_win_r = sum(r["r_multiple"] for r in wins)
    gross_loss_r = abs(sum(r["r_multiple"] for r in losses))
    if gross_loss_r > 0:
        profit_factor = gross_win_r / gross_loss_r
    else:
        profit_factor = None  # no losing trades - undefined, not "infinite edge"

    return {
        "total_fired": len(results), "resolved": total, "still_open": len(open_trades),
        "wins": len(wins), "losses": len(losses), "win_rate_pct": round(win_rate, 1),
        "avg_r_multiple": round(avg_r, 3), "total_r": round(total_r, 2),
        "max_drawdown_r": round(max_drawdown_r, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
    }


def breakdown_by(results: list, key: str) -> list:
    """
    Count / win-rate / avg-R per bucket of the given category, sorted by
    trade count descending. Buckets with very few trades are flagged, not
    hidden - a 100% win rate on 2 trades is noise, not a finding.
    """
    resolved = [r for r in results if r["outcome"] in ("win", "loss")]
    buckets = {}
    for r in resolved:
        buckets.setdefault(r.get(key, "unknown"), []).append(r)

    out = []
    for bucket, group in buckets.items():
        wins = sum(1 for r in group if r["outcome"] == "win")
        avg_r = sum(r["r_multiple"] for r in group) / len(group)
        out.append({
            "bucket": bucket, "count": len(group),
            "win_rate_pct": round(wins / len(group) * 100, 1),
            "avg_r": round(avg_r, 3),
            "low_sample": len(group) < 10,
        })
    return sorted(out, key=lambda x: -x["count"])


def format_category_report(results: list) -> str:
    categories = [
        ("session", "By session"), ("adx_bucket", "By ADX strength"),
        ("atr_pct_bucket", "By ATR percentile"), ("confidence_bucket", "By confidence band"),
        ("structure_basis", "By structure basis"), ("stop_basis", "By stop basis"), ("tp_basis", "By target basis"),
        ("direction", "By direction"),
    ]
    lines = ["\U0001F50D <b>Category breakdown</b>\n"]
    for key, label in categories:
        breakdown = breakdown_by(results, key)
        if not breakdown:
            continue
        lines.append(f"<b>{label}:</b>")
        for b in breakdown:
            flag = " \u26A0\uFE0F low sample" if b["low_sample"] else ""
            lines.append(f"  {b['bucket']}: n={b['count']}, win {b['win_rate_pct']}%, avg {b['avg_r']:+.2f}R{flag}")
        lines.append("")
    lines.append(
        "<i>Categories NOT included because this system has no detector for "
        "them yet: news events, a distinct mean-reversion regime, late-entry "
        "vs early-exit timing. Adding those would mean building new "
        "detection logic first, not just new reporting.</i>"
    )
    return "\n".join(lines)


def format_backtest_message(m5_df: pd.DataFrame, stats: dict) -> str:
    period_start = m5_df["time"].iloc[0].strftime("%Y-%m-%d")
    period_end = m5_df["time"].iloc[-1].strftime("%Y-%m-%d")
    if stats["resolved"] == 0:
        pf_display = "n/a (no resolved trades)"
    elif stats["profit_factor"] is not None:
        pf_display = f"{stats['profit_factor']}"
    else:
        pf_display = "undefined (zero losing trades)"

    return (
        f"\U0001F4C8 <b>{INSTRUMENT} Backtest Results</b>\n\n"
        f"Period: {period_start} to {period_end}\n"
        f"Signals fired: {stats['total_fired']} "
        f"({stats['resolved']} resolved, {stats['still_open']} still open)\n\n"
        f"Win rate: {stats['win_rate_pct']}%\n"
        f"Avg R per trade: {stats['avg_r_multiple']:+.2f}\n"
        f"Total R: {stats['total_r']:+.2f}\n"
        f"Max drawdown: {stats['max_drawdown_r']:.2f}R\n"
        f"Profit factor: {pf_display}\n\n"
        f"<i>Same scoring/risk logic as live bot.py, replayed against real "
        f"history - not a separate simulation. This validates the current "
        f"fixed thresholds/weights as a single baseline pass. It does not "
        f"yet include walk-forward re-splitting, Monte Carlo resampling, or "
        f"parameter sensitivity - those come next, once this result has "
        f"actually been reviewed.</i>"
    )


def main() -> None:
    print("Fetching historical M5 data...")
    m5_df = fetch_historical_paginated(TWELVEDATA_SYMBOL, "5min", BACKTEST_M5_TARGET_BARS, TWELVEDATA_API_KEY)
    print(f"Got {len(m5_df)} M5 bars: {m5_df['time'].iloc[0]} to {m5_df['time'].iloc[-1]}")

    print("Fetching historical H1 data...")
    h1_df = fetch_historical_paginated(TWELVEDATA_SYMBOL, "1h", BACKTEST_H1_TARGET_BARS, TWELVEDATA_API_KEY)
    print(f"Got {len(h1_df)} H1 bars: {h1_df['time'].iloc[0]} to {h1_df['time'].iloc[-1]}")

    print("Building feature frame...")
    frame = build_feature_frame(m5_df, h1_df)

    warmup = max(ATR_LOOKBACK, EMA_SLOW * 3, STRUCTURE_LOOKBACK + 1, ROC_LOOKBACK + 1, EMA_SLOPE_LOOKBACK + 1) + 10
    print(f"Running backtest from bar {warmup} of {len(frame)}...")
    results = run_backtest(frame, warmup_bars=warmup)

    stats = aggregate_results(results)
    print("Backtest complete:", stats)

    message = format_backtest_message(m5_df, stats)
    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message)

    if results:
        category_message = format_category_report(results)
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, category_message)


if __name__ == "__main__":
    main()