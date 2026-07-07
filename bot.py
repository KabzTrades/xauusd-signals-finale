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

def format_feature_message(price: float, features: dict) -> str:
    s = features["structure"]
    structure_line = "None detected"
    if s["bullish_reclaim"]:
        structure_line = f"Bullish reclaim (swept {s['prior_low']:.2f}, closed back above)"
    elif s["bearish_reclaim"]:
        structure_line = f"Bearish reclaim (swept {s['prior_high']:.2f}, closed back below)"
    elif s["swept_high"] or s["swept_low"]:
        structure_line = "Swept, no reclaim yet"

    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"\U0001F4CA <b>{INSTRUMENT} feature snapshot</b>\n\n"
        f"Price: {price:.2f}\n"
        f"Session: {features['session']}\n\n"
        f"ADX(H1): {features['adx_h1']}\n"
        f"Trend (M5 EMA{EMA_FAST}/{EMA_SLOW}): {features['trend_direction']} "
        f"(slope {features['trend_slope_pct']:+.2f}%)\n"
        f"HTF bias (H1): {features['htf_bias']}\n"
        f"ATR(M5): {features['atr_m5']} ({features['atr_percentile']}th percentile)\n"
        f"VWAP dev: {features['vwap_deviation_atr']:+.2f} ATR\n"
        f"ROC: {features['roc_pct']:+.2f}%\n"
        f"Structure: {structure_line}\n\n"
        f"Checked: {checked_at}\n\n"
        f"<i>Feature engine only - no scoring or threshold applied yet, "
        f"so this is not a trade signal.</i>"
    )


def main() -> None:
    price = get_current_price(TWELVEDATA_SYMBOL, TWELVEDATA_API_KEY)
    m5_df = get_time_series_df(TWELVEDATA_SYMBOL, "5min", M5_BAR_COUNT, TWELVEDATA_API_KEY)
    h1_df = get_time_series_df(TWELVEDATA_SYMBOL, "1h", H1_BAR_COUNT, TWELVEDATA_API_KEY)

    features = compute_features(m5_df, h1_df)
    message = format_feature_message(price, features)

    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message)
    print(f"Feature snapshot sent: {INSTRUMENT} @ {price} | {features}")


if __name__ == "__main__":
    main()