"""
XAUUSD Signal Bot - Connectivity Heartbeat (single-file version)

Everything - config, Twelve Data fetch, Telegram delivery - lives in this
one file on purpose. Fewer files means fewer manual "create file" steps
through GitHub's mobile editor, which is where every failure so far
actually happened, not in the logic itself. This is a packaging decision
for the mobile deployment path, not a shortcut on the trading system - the
same statistical rigor applies whether this is one file or nine.

NOT a trading signal yet - this proves the data feed and Telegram delivery
both work end-to-end. The real feature/scoring engine (Modules 3-4) plugs
in here later; main() is the seam where that happens.
"""

import os
import sys
import time
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Config - read straight from environment. GitHub Actions injects these from
# repo Secrets at run time; nothing here is typed into this file directly.
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


# ---------------------------------------------------------------------------
# Delivery: Telegram Bot API (raw HTTPS, no wrapper library)
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

        # Bad token, chat not found, malformed request - won't fix itself on retry
        raise RuntimeError(f"Telegram API rejected message: {last_error}")

    raise RuntimeError(f"Telegram send failed after {max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    price = get_current_price(TWELVEDATA_SYMBOL, TWELVEDATA_API_KEY)
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    message = (
        f"\u2705 <b>Bot online</b>\n\n"
        f"{INSTRUMENT} latest: {price:.2f}\n"
        f"Primary timeframe: {PRIMARY_TIMEFRAME}\n"
        f"Checked: {checked_at}\n\n"
        f"<i>Connectivity heartbeat only - the scoring engine isn't wired in "
        f"yet, so this is not a trade signal.</i>"
    )
    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message)
    print(f"Heartbeat sent: {INSTRUMENT} @ {price}")


if __name__ == "__main__":
    main()