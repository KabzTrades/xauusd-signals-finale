"""
XAUUSD Signal Bot - single-file version

Module 3 (Feature Engine) added on top of the working connectivity heartbeat.
Computes real indicators from real price data - ADX, ATR percentile, EMA
trend/slope, session VWAP, session tagging, liquidity-sweep structure,
momentum - and includes them in the Telegram message so you can verify the
math is actually running against live data.

STILL NOT a trading signal. Nothing here scores or combines these features
into a decision yet - that's Module 4. This module's only job is producing
honest numbers; Module 4's job is deciding what they're worth.

All periods/lookbacks below are starting defaults, not validated findings -
same principle as everywhere else in this build. Module 7 (walk-forward
validation against real historical data) is what actually earns the right
to call any of these numbers "correct" for this instrument/timeframe.
"""

import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        print(f"ERROR: missing required environment variable: {key}", file=sys.stderr)
        sys.exit(1)
    return value

TWELVEDATA_API_KEY = _require_env("TWELVEDATA_API_KEY")
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _require_env("TELEGRAM_CHAT_ID")

INSTRUMENT = "XAUUSD"
TWELVEDATA_SYMBOL = "XAU/USD"
PRIMARY_TIMEFRAME = "M5"
HTF_TIMEFRAME = "H1"

M5_BAR_COUNT = 500
H1_BAR_COUNT = 100

ATR_PERIOD = 14
ATR_LOOKBACK = 300
ADX_PERIOD = 14
EMA_FAST = 21
EMA_SLOW = 50
EMA_SLOPE_LOOKBACK = 5
STRUCTURE_LOOKBACK = 20
ROC_LOOKBACK = 10

SESSION_WINDOWS = {
    "asia_start": 0, "asia_end": 7,
    "london_open_start": 7, "london_open_end": 12,
    "overlap_start": 12, "overlap_end": 16,
    "ny_afternoon_start": 16, "ny_afternoon_end": 21,
}

# ---------------------------------------------------------------------------
# Module 4 - Signal Scoring Engine constants
#
# Every number below is a STARTING DEFAULT, not a validated finding - same
# principle as everywhere else in this build. These get replaced with real
# numbers once Module 7 (walk-forward validation against actual historical
# data) has something to say about them. Nothing here should be read as
# "the edge" - it's the scaffolding the edge gets measured against.
# ---------------------------------------------------------------------------

# Hard gates - any failure means no signal at all, regardless of score.
ADX_GATE_FLOOR = 22.0             # ADX(H1) below this -> regime not trending, no signal
ATR_PCT_GATE_FLOOR = 30.0         # ATR percentile below this -> too quiet/chop, suppress
ATR_PCT_GATE_CEILING = 95.0       # ATR percentile above this -> likely news-spike/illiquid, suppress

# Continuous scoring scales (0-100), only evaluated once gates pass.
ADX_SCORE_ZERO = 15.0             # ADX at/below this -> regime score 0
ADX_SCORE_FULL = 40.0             # ADX at/above this -> regime score 100
TREND_SLOPE_SCORE_CAP = 0.30      # abs(EMA slope %) at/above this -> trend score 100
MOMENTUM_ROC_SCORE_CAP = 0.50     # signed ROC % at/above this -> momentum score 100
ATR_PCT_SCORE_PEAK_LOW = 50.0     # volatility score = 100 within this band -
ATR_PCT_SCORE_PEAK_HIGH = 80.0    # constructive expansion, not chop, not a spike

SESSION_SCORES = {
    "London/NY Overlap": 100.0,
    "London Open": 80.0,
    "NY Afternoon": 60.0,
    "Asia": 30.0,
    "Off Hours": 10.0,
}

# Composite weights - sum to 1.0. Trend and Structure weighted highest since
# they carry the most direct directional evidence; Regime/Momentum next;
# Volatility/Session are context modifiers. Reweighted in Module 7 based on
# which components actually predict out-of-sample outcomes, not intuition.
SCORE_WEIGHTS = {
    "regime": 0.15,
    "trend": 0.25,
    "momentum": 0.15,
    "volatility": 0.10,
    "structure": 0.25,
    "session": 0.10,
}

MIN_RISK_REWARD = 1.5

# ---------------------------------------------------------------------------
# Module 5 - Risk Engine constants
#
# Replaces the flat ATR-multiple stop from Module 4 with structure-aware
# placement: the stop goes beyond the actual rolling structure level (where
# the trade thesis is genuinely invalidated), not an arbitrary distance.
# Floor/ceiling exist because structure can sit unrealistically close (noise-
# prone stop) or unrealistically far (undefined risk) from current price -
# both need bounding, not blind trust in the raw structure level.
# ---------------------------------------------------------------------------
MIN_STOP_ATR_MULTIPLE = 0.5       # stop distance floor - structure closer than this is too tight to trust
MAX_STOP_ATR_MULTIPLE = 3.0       # stop distance ceiling - structure farther than this means undefined risk
STRUCTURE_STOP_BUFFER_ATR = 0.25  # buffer beyond the structural level, in ATR units - avoids stop-hunting noise

# Fixed hard floor for now. The taper mechanism discussed earlier (bounded
# floor-drift toward a daily signal-count target) needs state persisted
# across runs - GitHub Actions gives each run a clean slate, so there's no
# "how many signals fired today" to taper against yet. That's Module 9
# (performance tracking / signal journal). Until then: fixed floor, no quota.
PUBLISH_FLOOR = 78.0


# ---------------------------------------------------------------------------
# Data source: Twelve Data free REST API
# ---------------------------------------------------------------------------

def get_current_price(symbol: str, api_key: str, max_retries: int = 3) -> float:
    url = "https://api.twelvedata.com/price"
    params = {"symbol": symbol, "apikey": api_key}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            time.sleep(5 * attempt)
            continue

        if isinstance(data, dict) and data.get("status") == "error":
            last_error = data.get("message", "unknown error")
            rate_limited = resp.status_code == 429 or "limit" in str(last_error).lower()
            if rate_limited and attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            raise RuntimeError(f"Twelve Data API error: {last_error}")

        if "price" not in data:
            raise RuntimeError(f"No price in Twelve Data response: {data}")

        return float(data["price"])

    raise RuntimeError(f"Twelve Data request failed after {max_retries} attempts: {last_error}")


def get_time_series_df(symbol: str, interval: str, output_size: int, api_key: str, max_retries: int = 3) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": output_size, "timezone": "UTC", "apikey": api_key}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            time.sleep(5 * attempt)
            continue

        if isinstance(data, dict) and data.get("status") == "error":
            last_error = data.get("message", "unknown error")
            rate_limited = resp.status_code == 429 or "limit" in str(last_error).lower()
            if rate_limited and attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            raise RuntimeError(f"Twelve Data API error ({symbol} {interval}): {last_error}")

        values = data.get("values")
        if not values:
            raise RuntimeError(f"No bars returned for {symbol} {interval}: {data}")

        df = pd.DataFrame(values)
        df["time"] = pd.to_datetime(df["datetime"], utc=True)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        df = df.sort_values("time").reset_index(drop=True)
        return df[["time", "open", "high", "low", "close"]]

    raise RuntimeError(f"Twelve Data request failed after {max_retries} attempts ({symbol} {interval}): {last_error}")


# ---------------------------------------------------------------------------
# Feature engine (Module 3)
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder's ATR - ewm(alpha=1/period) is the standard equivalent of Wilder's smoothing."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_percentile_rank(atr_series: pd.Series, lookback: int = ATR_LOOKBACK) -> float:
    recent = atr_series.dropna().tail(lookback)
    if len(recent) < max(20, lookback // 3):
        raise RuntimeError(f"Insufficient ATR history for percentile rank: {len(recent)} bars")
    current = recent.iloc[-1]
    return float((recent < current).sum()) / len(recent) * 100


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """Wilder's ADX. Same ewm(alpha=1/period) smoothing applied to +DI/-DI and to DX itself."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def ema_slope_pct(ema_series: pd.Series, lookback: int = EMA_SLOPE_LOOKBACK) -> float:
    """% change of the EMA over `lookback` bars - normalized so it's comparable across price levels."""
    recent = ema_series.dropna()
    if len(recent) < lookback + 1:
        raise RuntimeError("Insufficient EMA history for slope calculation")
    prior = recent.iloc[-1 - lookback]
    return (recent.iloc[-1] - prior) / prior * 100


def compute_htf_bias(htf_df: pd.DataFrame, ema_period: int = EMA_SLOW) -> str:
    ema = compute_ema(htf_df["close"], ema_period)
    if pd.isna(ema.iloc[-1]):
        raise RuntimeError("Insufficient HTF history for bias EMA")
    return "Bullish" if htf_df["close"].iloc[-1] > ema.iloc[-1] else "Bearish"


def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Anchored to UTC calendar day. Note: this feed doesn't provide reliable
    volume for gold, so this is a typical-price TWAP (time-weighted), not a
    true volume-weighted VWAP. Serves the same fair-value-anchor role;
    labeled honestly rather than fabricating volume weighting we don't have.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    day = df["time"].dt.date
    cum_tp = typical_price.groupby(day).cumsum()
    cum_count = typical_price.groupby(day).cumcount() + 1
    return cum_tp / cum_count


def get_session(hour_utc: int, windows: dict = SESSION_WINDOWS) -> str:
    if windows["asia_start"] <= hour_utc < windows["asia_end"]:
        return "Asia"
    if windows["london_open_start"] <= hour_utc < windows["london_open_end"]:
        return "London Open"
    if windows["overlap_start"] <= hour_utc < windows["overlap_end"]:
        return "London/NY Overlap"
    if windows["ny_afternoon_start"] <= hour_utc < windows["ny_afternoon_end"]:
        return "NY Afternoon"
    return "Off Hours"


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = STRUCTURE_LOOKBACK) -> dict:
    """
    Sweep = current bar's high/low exceeds the prior `lookback` bars' range.
    Reclaim = close comes back inside that prior range in the same bar -
    the displacement-then-reclaim pattern flagged in research as having real
    statistical follow-through, vs. a bare wick with no reclaim.
    """
    if len(df) < lookback + 1:
        raise RuntimeError("Insufficient bars for structure detection")

    prior = df.iloc[-(lookback + 1):-1]
    current = df.iloc[-1]
    prior_high, prior_low = prior["high"].max(), prior["low"].min()

    swept_high = current["high"] > prior_high
    swept_low = current["low"] < prior_low

    return {
        "swept_high": bool(swept_high),
        "swept_low": bool(swept_low),
        "bullish_reclaim": bool(swept_low and current["close"] > prior_low),
        "bearish_reclaim": bool(swept_high and current["close"] < prior_high),
        "prior_high": float(prior_high),
        "prior_low": float(prior_low),
    }


def compute_roc(close_series: pd.Series, lookback: int = ROC_LOOKBACK) -> float:
    recent = close_series.dropna()
    if len(recent) < lookback + 1:
        raise RuntimeError("Insufficient bars for ROC calculation")
    prior = recent.iloc[-1 - lookback]
    return (recent.iloc[-1] - prior) / prior * 100


def compute_features(m5_df: pd.DataFrame, h1_df: pd.DataFrame) -> dict:
    """Runs the full feature engine against fresh M5 + H1 data. Raises loudly on insufficient history."""
    atr_series = compute_atr(m5_df)
    latest_atr = atr_series.dropna().iloc[-1]
    atr_pct = atr_percentile_rank(atr_series)

    adx_series = compute_adx(h1_df)
    latest_adx = adx_series.dropna().iloc[-1]

    ema_fast = compute_ema(m5_df["close"], EMA_FAST)
    ema_slow = compute_ema(m5_df["close"], EMA_SLOW)
    trend_direction = "Bullish" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "Bearish"
    trend_slope = ema_slope_pct(ema_fast)

    htf_bias = compute_htf_bias(h1_df)

    vwap_series = compute_session_vwap(m5_df)
    latest_close = m5_df["close"].iloc[-1]
    vwap_dev_atr = (latest_close - vwap_series.iloc[-1]) / latest_atr

    session = get_session(m5_df["time"].iloc[-1].hour)
    structure = detect_liquidity_sweep(m5_df)
    roc = compute_roc(m5_df["close"])

    return {
        "atr_m5": round(float(latest_atr), 3),
        "atr_percentile": round(atr_pct, 1),
        "adx_h1": round(float(latest_adx), 1),
        "trend_direction": trend_direction,
        "trend_slope_pct": round(trend_slope, 3),
        "htf_bias": htf_bias,
        "vwap": round(float(vwap_series.iloc[-1]), 2),
        "vwap_deviation_atr": round(float(vwap_dev_atr), 2),
        "session": session,
        "roc_pct": round(roc, 3),
        "structure": structure,
    }


# ---------------------------------------------------------------------------
# Signal scoring engine (Module 4)
#
# Regime is a hard gate, not a score - no signal is even evaluated below the
# ADX/ATR floors. Direction requires M5 trend and H1 bias to agree - a
# conflict between them means no candidate at all, not a low-confidence one.
# Only once both pass does the composite score get computed.
# ---------------------------------------------------------------------------

def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def determine_candidate_direction(features: dict):
    trend_dir = features["trend_direction"]
    htf_dir = features["htf_bias"]
    if trend_dir != htf_dir:
        return None
    return "BUY" if trend_dir == "Bullish" else "SELL"


def check_regime_gate(features: dict):
    adx = features["adx_h1"]
    atr_pct = features["atr_percentile"]
    if adx < ADX_GATE_FLOOR:
        return False, f"ADX(H1) {adx} below trend floor {ADX_GATE_FLOOR}"
    if atr_pct < ATR_PCT_GATE_FLOOR:
        return False, f"ATR percentile {atr_pct} below floor {ATR_PCT_GATE_FLOOR} (chop)"
    if atr_pct > ATR_PCT_GATE_CEILING:
        return False, f"ATR percentile {atr_pct} above ceiling {ATR_PCT_GATE_CEILING} (illiquid/news-spike)"
    return True, "regime gate passed"


def score_regime(adx_h1: float) -> float:
    if adx_h1 <= ADX_SCORE_ZERO:
        return 0.0
    if adx_h1 >= ADX_SCORE_FULL:
        return 100.0
    return _clip((adx_h1 - ADX_SCORE_ZERO) / (ADX_SCORE_FULL - ADX_SCORE_ZERO) * 100)


def score_trend(slope_pct: float) -> float:
    return _clip(abs(slope_pct) / TREND_SLOPE_SCORE_CAP * 100)


def score_momentum(roc_pct: float, direction: str) -> float:
    """Only rewards momentum agreeing with the candidate direction; scores 0 if it opposes."""
    signed = roc_pct if direction == "BUY" else -roc_pct
    if signed <= 0:
        return 0.0
    return _clip(signed / MOMENTUM_ROC_SCORE_CAP * 100)


def score_volatility(atr_percentile: float) -> float:
    """Triangular peak: constructive expansion scores highest, tapering toward both gate edges."""
    if ATR_PCT_SCORE_PEAK_LOW <= atr_percentile <= ATR_PCT_SCORE_PEAK_HIGH:
        return 100.0
    if atr_percentile < ATR_PCT_SCORE_PEAK_LOW:
        span = ATR_PCT_SCORE_PEAK_LOW - ATR_PCT_GATE_FLOOR
        return _clip((atr_percentile - ATR_PCT_GATE_FLOOR) / span * 100) if span > 0 else 100.0
    span = ATR_PCT_GATE_CEILING - ATR_PCT_SCORE_PEAK_HIGH
    return _clip((ATR_PCT_GATE_CEILING - atr_percentile) / span * 100) if span > 0 else 100.0


def score_structure(structure: dict, direction: str) -> float:
    agreeing_reclaim = structure["bullish_reclaim"] if direction == "BUY" else structure["bearish_reclaim"]
    opposing_reclaim = structure["bearish_reclaim"] if direction == "BUY" else structure["bullish_reclaim"]
    if agreeing_reclaim:
        return 100.0
    if opposing_reclaim:
        return 0.0
    if structure["swept_high"] or structure["swept_low"]:
        return 40.0
    return 50.0


def score_session(session: str) -> float:
    return SESSION_SCORES.get(session, 10.0)


def compute_composite_score(features: dict, direction: str) -> dict:
    scores = {
        "regime": score_regime(features["adx_h1"]),
        "trend": score_trend(features["trend_slope_pct"]),
        "momentum": score_momentum(features["roc_pct"], direction),
        "volatility": score_volatility(features["atr_percentile"]),
        "structure": score_structure(features["structure"], direction),
        "session": score_session(features["session"]),
    }
    composite = sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    return {"breakdown": scores, "composite": composite}


def compute_risk_levels(price: float, atr: float, direction: str, structure: dict) -> dict:
    """
    Module 5 - Risk Engine. Stop is placed beyond the actual rolling
    structure level (prior 20-bar high/low from Module 3), not an arbitrary
    ATR multiple - buffered against noise, and bounded to a sane ATR range
    so risk is never undefined-thin or unrealistically wide.

    Take-profit uses the sweep's own measured range (the size of the
    manipulated range, projected onward) when a genuine reclaim triggered
    the signal - otherwise falls back to the minimum validated R:R. This is
    what makes R:R vary signal to signal instead of sitting fixed at exactly
    MIN_RISK_REWARD every time, which was Module 4's stated limitation.
    """
    buffer = atr * STRUCTURE_STOP_BUFFER_ATR
    floor_distance = MIN_STOP_ATR_MULTIPLE * atr
    ceiling_distance = MAX_STOP_ATR_MULTIPLE * atr

    if direction == "BUY":
        raw_stop_distance = price - (structure["prior_low"] - buffer)
    else:
        raw_stop_distance = (structure["prior_high"] + buffer) - price

    if raw_stop_distance < floor_distance:
        sl_distance = floor_distance
        stop_basis = "ATR floor (structure too close for a realistic stop)"
    elif raw_stop_distance > ceiling_distance:
        sl_distance = ceiling_distance
        stop_basis = "ATR ceiling (structure too far for defined risk)"
    else:
        sl_distance = raw_stop_distance
        stop_basis = "structure level"

    reclaimed = structure["bullish_reclaim"] if direction == "BUY" else structure["bearish_reclaim"]
    measured_move = (structure["prior_high"] - structure["prior_low"]) if reclaimed else 0.0
    min_rr_distance = sl_distance * MIN_RISK_REWARD

    if measured_move > min_rr_distance:
        tp_distance = measured_move
        tp_basis = "measured move (sweep range)"
    else:
        tp_distance = min_rr_distance
        tp_basis = f"minimum {MIN_RISK_REWARD}:1 R:R"

    if direction == "BUY":
        stop_loss = price - sl_distance
        take_profit = price + tp_distance
    else:
        stop_loss = price + sl_distance
        take_profit = price - tp_distance

    return {
        "entry": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "stop_basis": stop_basis,
        "tp_basis": tp_basis,
    }


def evaluate_signal(price: float, features: dict) -> dict:
    """
    Always returns a result describing what happened, whether or not a
    signal actually fires - used both to publish real signals and to show
    diagnostic scoring during this build/validation phase. Once the system
    is validated, production behavior goes silent on non-fires; showing the
    reasoning now is deliberate, for verifying the engine before trusting it.
    """
    direction = determine_candidate_direction(features)
    if direction is None:
        return {"fired": False, "direction": None, "confidence": None, "breakdown": None,
                "reason": "M5/H1 trend conflict - no directional candidate"}

    gate_ok, gate_reason = check_regime_gate(features)
    if not gate_ok:
        return {"fired": False, "direction": direction, "confidence": None, "breakdown": None,
                "reason": gate_reason}

    scoring = compute_composite_score(features, direction)
    levels = compute_risk_levels(price, features["atr_m5"], direction, features["structure"])
    risk_reward = levels["tp_distance"] / levels["sl_distance"] if levels["sl_distance"] > 0 else 0.0

    fired = scoring["composite"] >= PUBLISH_FLOOR
    return {
        "fired": fired,
        "direction": direction,
        "confidence": round(scoring["composite"], 1),
        "breakdown": {k: round(v, 1) for k, v in scoring["breakdown"].items()},
        "levels": levels,
        "risk_reward": round(risk_reward, 2),
        "reason": None if fired else f"confidence {scoring['composite']:.1f} below publish floor {PUBLISH_FLOOR}",
    }


# ---------------------------------------------------------------------------
# Delivery: Telegram Bot API
# ---------------------------------------------------------------------------

def send_telegram_message(token: str, chat_id: str, text: str, max_retries: int = 3) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            time.sleep(3 * attempt)
            continue

        if body.get("ok"):
            print(f"Telegram message delivered (message_id={body['result']['message_id']})")
            return

        last_error = body.get("description", "unknown Telegram API error")
        if resp.status_code == 429:
            retry_after = body.get("parameters", {}).get("retry_after", 3 * attempt)
            time.sleep(retry_after)
            continue

        raise RuntimeError(f"Telegram API rejected message: {last_error}")

    raise RuntimeError(f"Telegram send failed after {max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_reason(direction: str, features: dict) -> str:
    direction_word = "bullish" if direction == "BUY" else "bearish"
    bits = [
        f"H1 ADX {features['adx_h1']} confirms trending regime",
        f"M5/H1 trend aligned {direction_word} (slope {features['trend_slope_pct']:+.2f}%)",
    ]
    s = features["structure"]
    if (direction == "BUY" and s["bullish_reclaim"]) or (direction == "SELL" and s["bearish_reclaim"]):
        bits.append("confirmed by liquidity-sweep reclaim")
    bits.append(f"{features['session']} session")
    return "; ".join(bits) + "."


def format_signal_message(evaluation: dict, features: dict) -> str:
    direction = evaluation["direction"]
    levels = evaluation["levels"]
    b = evaluation["breakdown"]
    emoji = "\U0001F680" if direction == "BUY" else "\U0001F53B"
    invalidation_dir = "below" if direction == "BUY" else "above"
    htf_flip = "bearish" if direction == "BUY" else "bullish"
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"{emoji} <b>{direction} {INSTRUMENT}</b>\n\n"
        f"Entry: {levels['entry']:.2f}\n"
        f"Stop Loss: {levels['stop_loss']:.2f}\n"
        f"Take Profit: {levels['take_profit']:.2f}\n"
        f"Risk Reward: 1:{evaluation['risk_reward']:.1f}\n"
        f"Confidence: {evaluation['confidence']:.0f}%\n\n"
        f"Market Regime:\nTrending\n\n"
        f"Reason:\n{_build_reason(direction, features)}\n\n"
        f"Invalidation:\nClose back {invalidation_dir} {levels['stop_loss']:.2f} on M5, "
        f"or H1 bias flips {htf_flip}\n\n"
        f"<i>Regime {b['regime']:.0f} / Trend {b['trend']:.0f} / Momentum {b['momentum']:.0f} / "
        f"Volatility {b['volatility']:.0f} / Structure {b['structure']:.0f} / Session {b['session']:.0f}</i>\n"
        f"<i>SL basis: {levels['stop_basis']} | TP basis: {levels['tp_basis']}</i>\n\n"
        f"<i>{checked_at}</i>"
    )


def format_no_signal_message(price: float, evaluation: dict, features: dict) -> str:
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"\u26AA <b>{INSTRUMENT} - no signal</b>\n",
        f"Price: {price:.2f}",
        f"Session: {features['session']}",
        f"Reason: {evaluation['reason']}",
    ]
    if evaluation["breakdown"] is not None:
        b = evaluation["breakdown"]
        lines.append(
            f"\nRegime {b['regime']:.0f} / Trend {b['trend']:.0f} / Momentum {b['momentum']:.0f} / "
            f"Volatility {b['volatility']:.0f} / Structure {b['structure']:.0f} / Session {b['session']:.0f}"
        )
    lines.append(f"\nChecked: {checked_at}")
    lines.append(
        "\n<i>Diagnostic output during validation - once calibrated, "
        "production stays silent on no-signal runs instead of messaging every cycle.</i>"
    )
    return "\n".join(lines)


def main() -> None:
    price = get_current_price(TWELVEDATA_SYMBOL, TWELVEDATA_API_KEY)
    m5_df = get_time_series_df(TWELVEDATA_SYMBOL, "5min", M5_BAR_COUNT, TWELVEDATA_API_KEY)
    h1_df = get_time_series_df(TWELVEDATA_SYMBOL, "1h", H1_BAR_COUNT, TWELVEDATA_API_KEY)

    features = compute_features(m5_df, h1_df)
    evaluation = evaluate_signal(price, features)

    if evaluation["fired"]:
        message = format_signal_message(evaluation, features)
    else:
        message = format_no_signal_message(price, evaluation, features)

    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message)
    print(f"Evaluated: {INSTRUMENT} @ {price} | fired={evaluation['fired']} | {evaluation}")


if __name__ == "__main__":
    main()