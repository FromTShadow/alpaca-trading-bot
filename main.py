import os
import sys
import csv
import logging
import threading
import time as time_module
from datetime import datetime, date, time, timedelta
from pathlib import Path

import pytz
import schedule
import pandas as pd
from flask import Flask, jsonify, render_template
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
# Set to False when you are ready to trade with real money.
# WARNING: False will place REAL orders against your live brokerage account.
PAPER_MODE = True

SYMBOLS         = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
MAX_DAILY_SPEND = 80.0
STOP_LOSS_PCT   = 0.02
TAKE_PROFIT_PCT = 0.04
SHORT_MA        = 9
LONG_MA         = 21
MARKET_OPEN     = time(9, 30)
MARKET_CLOSE    = time(16, 0)
ANALYSIS_TIME   = time(16, 5)
EST             = pytz.timezone("America/New_York")
TIMEFRAME       = TimeFrame(5, TimeFrameUnit.Minute)
TRADE_LOG_FILE  = "trade_log.csv"
MIN_WEIGHT      = 0.05
MAX_WEIGHT      = 0.40

CSV_HEADERS = [
    "date", "symbol", "qty",
    "buy_time",  "buy_price",  "ma_short_at_buy",  "ma_long_at_buy",
    "sell_time", "sell_price", "ma_short_at_sell", "ma_long_at_sell",
    "profit_loss", "profit_loss_pct", "exit_reason",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)   # silence Flask request noise

# ── Alpaca clients ────────────────────────────────────────────────────────────
api_key    = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET_KEY"]

trading_client = TradingClient(api_key, secret_key, paper=PAPER_MODE)
data_client    = StockHistoricalDataClient(api_key, secret_key)

# ── Shared state (protected by state_lock) ────────────────────────────────────
state_lock         = threading.Lock()
daily_spend        = 0.0
last_reset_date    = None
last_analysis_date = None
open_trades: dict  = {}                                        # symbol -> entry dict
symbol_weights     = {s: 1.0 / len(SYMBOLS) for s in SYMBOLS}


# ── CSV helpers ───────────────────────────────────────────────────────────────
def ensure_csv_exists() -> None:
    if not Path(TRADE_LOG_FILE).exists():
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
        logger.info(f"Created {TRADE_LOG_FILE}.")


def append_trade(record: dict) -> None:
    ensure_csv_exists()
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(record)
    logger.info(f"Trade logged to {TRADE_LOG_FILE}.")


def build_trade_record(
    symbol, qty, buy_time, buy_price, ma_short_buy, ma_long_buy,
    sell_time, sell_price, ma_short_sell, ma_long_sell, exit_reason,
) -> dict:
    pl     = (sell_price - buy_price) * qty
    pl_pct = (sell_price - buy_price) / buy_price * 100
    return {
        "date":              datetime.now(EST).strftime("%Y-%m-%d"),
        "symbol":            symbol,
        "qty":               qty,
        "buy_time":          buy_time,
        "buy_price":         round(buy_price, 4),
        "ma_short_at_buy":   round(ma_short_buy, 4),
        "ma_long_at_buy":    round(ma_long_buy, 4),
        "sell_time":         sell_time,
        "sell_price":        round(sell_price, 4),
        "ma_short_at_sell":  round(ma_short_sell, 4) if ma_short_sell != "" else "",
        "ma_long_at_sell":   round(ma_long_sell, 4)  if ma_long_sell  != "" else "",
        "profit_loss":       round(pl, 4),
        "profit_loss_pct":   round(pl_pct, 4),
        "exit_reason":       exit_reason,
    }


# ── Market helpers ────────────────────────────────────────────────────────────
def is_market_open() -> bool:
    now = datetime.now(EST)
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < MARKET_CLOSE


def reset_daily_spend_if_needed() -> None:
    global daily_spend, last_reset_date
    today = datetime.now(EST).date()
    with state_lock:
        if last_reset_date != today:
            daily_spend     = 0.0
            last_reset_date = today
            logger.info(f"Daily spend counter reset for {today}.")


# ── Alpaca data helpers ───────────────────────────────────────────────────────
def get_bars(symbol: str) -> pd.DataFrame:
    end   = datetime.now(EST)
    start = end - timedelta(hours=6)
    req   = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        limit=60,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(req)
    df   = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.loc[symbol]
    return df.reset_index(drop=True)


def get_position(symbol: str):
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


def place_bracket_order(symbol: str, qty: int, price: float):
    stop_price        = round(price * (1 - STOP_LOSS_PCT),   2)
    take_profit_price = round(price * (1 + TAKE_PROFIT_PCT), 2)
    return trading_client.submit_order(MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class="bracket",
        stop_loss=StopLossRequest(stop_price=stop_price),
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
    ))


# ── SL / TP exit detection ────────────────────────────────────────────────────
def check_closed_positions() -> None:
    with state_lock:
        symbols_to_check = list(open_trades.keys())

    for symbol in symbols_to_check:
        if get_position(symbol) is not None:
            continue

        with state_lock:
            entry = open_trades.pop(symbol, None)
        if entry is None:
            continue

        sell_price  = entry["buy_price"]
        sell_time   = datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S")
        exit_reason = "sl_or_tp"

        try:
            orders = trading_client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                limit=20,
            ))
            for o in orders:
                if o.side == OrderSide.SELL and o.filled_avg_price:
                    sell_price  = float(o.filled_avg_price)
                    sell_time   = (
                        o.filled_at.astimezone(EST).strftime("%Y-%m-%d %H:%M:%S")
                        if o.filled_at else sell_time
                    )
                    tp_thresh = entry["buy_price"] * (1 + TAKE_PROFIT_PCT * 0.90)
                    sl_thresh = entry["buy_price"] * (1 - STOP_LOSS_PCT  * 0.90)
                    exit_reason = (
                        "take_profit" if sell_price >= tp_thresh else
                        "stop_loss"   if sell_price <= sl_thresh else
                        "sl_or_tp"
                    )
                    break
        except Exception as e:
            logger.error(f"{symbol}: Could not fetch closing order — {e}")

        record = build_trade_record(
            symbol=symbol, qty=entry["qty"],
            buy_time=entry["buy_time"], buy_price=entry["buy_price"],
            ma_short_buy=entry["ma_short"], ma_long_buy=entry["ma_long"],
            sell_time=sell_time, sell_price=sell_price,
            ma_short_sell="", ma_long_sell="",
            exit_reason=exit_reason,
        )
        append_trade(record)
        pl     = (sell_price - entry["buy_price"]) * entry["qty"]
        pl_pct = (sell_price - entry["buy_price"]) / entry["buy_price"] * 100
        logger.info(
            f"{symbol}: Closed via {exit_reason}. "
            f"${entry['buy_price']:.2f} → ${sell_price:.2f} | "
            f"P&L ${pl:.2f} ({pl_pct:.2f}%)"
        )


# ── End-of-day analysis ───────────────────────────────────────────────────────
def run_end_of_day_analysis() -> None:
    global symbol_weights

    logger.info("=" * 60)
    logger.info("END-OF-DAY ANALYSIS")
    logger.info("=" * 60)

    if not Path(TRADE_LOG_FILE).exists():
        logger.info("No trade log yet. Keeping equal weights.")
        return

    try:
        df = pd.read_csv(TRADE_LOG_FILE)
    except Exception as e:
        logger.error(f"Could not read {TRADE_LOG_FILE}: {e}")
        return

    if df.empty:
        logger.info("Trade log is empty. Keeping equal weights.")
        return

    df["profit_loss"]     = pd.to_numeric(df["profit_loss"],     errors="coerce")
    df["profit_loss_pct"] = pd.to_numeric(df["profit_loss_pct"], errors="coerce")

    logger.info(f"Total trades: {len(df)} across {df['symbol'].nunique()} symbol(s).")
    logger.info(f"{'Symbol':<6} {'Trades':>6} {'Win%':>6} {'Avg P&L%':>9} {'Total P&L':>10}")
    logger.info("-" * 42)

    scores = {}
    for symbol in SYMBOLS:
        sdf = df[df["symbol"] == symbol]
        if sdf.empty:
            logger.info(f"{symbol:<6} {'0':>6}  — no history, neutral score")
            scores[symbol] = 0.25
            continue
        wins     = (sdf["profit_loss"] > 0).sum()
        total    = len(sdf)
        win_rate = wins / total
        avg_pl   = sdf["profit_loss_pct"].mean()
        total_pl = sdf["profit_loss"].sum()
        logger.info(f"{symbol:<6} {total:>6} {win_rate*100:>5.1f}% {avg_pl:>8.2f}% {total_pl:>9.2f}")
        scores[symbol] = (win_rate * 0.5) + (avg_pl * 0.5)

    min_score = min(scores.values())
    if min_score < 0.01:
        shift  = abs(min_score) + 0.01
        scores = {k: v + shift for k, v in scores.items()}

    total_score = sum(scores.values())
    raw         = {k: v / total_score for k, v in scores.items()}
    clipped     = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for k, v in raw.items()}
    total_clip  = sum(clipped.values())

    new_weights = {k: v / total_clip for k, v in clipped.items()}
    with state_lock:
        symbol_weights = new_weights

    logger.info("\nUpdated budget allocations for tomorrow:")
    logger.info(f"{'Symbol':<6} {'Weight%':>8} {'Budget $':>9}")
    logger.info("-" * 28)
    for symbol in SYMBOLS:
        w = symbol_weights[symbol]
        logger.info(f"{symbol:<6} {w*100:>7.1f}%  ${MAX_DAILY_SPEND * w:>7.2f}")
    logger.info("=" * 60)


def maybe_run_end_of_day_analysis() -> None:
    global last_analysis_date
    now   = datetime.now(EST)
    today = now.date()
    if now.time() >= ANALYSIS_TIME and last_analysis_date != today and now.weekday() < 5:
        last_analysis_date = today
        run_end_of_day_analysis()


# ── Trading strategy ──────────────────────────────────────────────────────────
def run_strategy() -> None:
    global daily_spend

    reset_daily_spend_if_needed()

    if not is_market_open():
        logger.info("Market is closed. Skipping run.")
        return

    check_closed_positions()

    with state_lock:
        spent = daily_spend
        weights = dict(symbol_weights)

    logger.info(f"--- Strategy run | Daily spend: ${spent:.2f} / ${MAX_DAILY_SPEND:.2f} ---")

    for symbol in SYMBOLS:
        try:
            df = get_bars(symbol)
            if len(df) < LONG_MA + 1:
                logger.warning(f"{symbol}: Only {len(df)} bars — need {LONG_MA+1}. Skipping.")
                continue

            df["ma_short"] = df["close"].rolling(SHORT_MA).mean()
            df["ma_long"]  = df["close"].rolling(LONG_MA).mean()

            latest    = df.iloc[-1]
            prev      = df.iloc[-2]
            ma_s_now  = float(latest["ma_short"])
            ma_l_now  = float(latest["ma_long"])
            ma_s_prev = float(prev["ma_short"])
            ma_l_prev = float(prev["ma_long"])
            price     = float(latest["close"])

            if pd.isna(ma_s_now) or pd.isna(ma_l_now):
                logger.warning(f"{symbol}: NaN MA values. Skipping.")
                continue

            crossed_above = (ma_s_now > ma_l_now) and (ma_s_prev <= ma_l_prev)
            crossed_below = (ma_s_now < ma_l_now) and (ma_s_prev >= ma_l_prev)
            position      = get_position(symbol)

            with state_lock:
                in_trade = symbol in open_trades

            # BUY
            if crossed_above:
                if position is not None or in_trade:
                    logger.info(f"{symbol}: Bullish crossover — already in position. MA9={ma_s_now:.2f} MA21={ma_l_now:.2f}")
                else:
                    weight      = weights.get(symbol, 1 / len(SYMBOLS))
                    sym_budget  = MAX_DAILY_SPEND * weight
                    with state_lock:
                        budget_left = min(sym_budget, MAX_DAILY_SPEND - daily_spend)

                    if budget_left <= 0:
                        logger.info(f"{symbol}: Bullish crossover — budget exhausted ({weight*100:.1f}% alloc). Skipping.")
                        continue

                    qty  = max(1, int(budget_left // price))
                    cost = qty * price

                    if cost > budget_left:
                        logger.info(f"{symbol}: Bullish crossover — not enough budget (need ${cost:.2f}, have ${budget_left:.2f}). Skipping.")
                        continue

                    now_str = datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(
                        f"{symbol}: BUY {qty} @ ~${price:.2f} (cost ~${cost:.2f}, "
                        f"alloc {weight*100:.1f}%) | MA9={ma_s_now:.2f} MA21={ma_l_now:.2f}"
                    )
                    order = place_bracket_order(symbol, qty, price)

                    with state_lock:
                        daily_spend += cost
                        open_trades[symbol] = {
                            "buy_time":  now_str,
                            "buy_price": price,
                            "qty":       qty,
                            "ma_short":  ma_s_now,
                            "ma_long":   ma_l_now,
                            "order_id":  str(order.id),
                        }

                    logger.info(
                        f"{symbol}: Order {order.id} placed. "
                        f"SL=${price*(1-STOP_LOSS_PCT):.2f}  TP=${price*(1+TAKE_PROFIT_PCT):.2f}  "
                        f"Daily spend ${daily_spend:.2f}"
                    )

            # SELL
            elif crossed_below:
                if position is not None or in_trade:
                    logger.info(f"{symbol}: Bearish crossover — closing. MA9={ma_s_now:.2f} MA21={ma_l_now:.2f}")
                    trading_client.close_position(symbol)

                    with state_lock:
                        entry = open_trades.pop(symbol, None)

                    if entry:
                        now_str = datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S")
                        record  = build_trade_record(
                            symbol=symbol, qty=entry["qty"],
                            buy_time=entry["buy_time"], buy_price=entry["buy_price"],
                            ma_short_buy=entry["ma_short"], ma_long_buy=entry["ma_long"],
                            sell_time=now_str, sell_price=price,
                            ma_short_sell=ma_s_now, ma_long_sell=ma_l_now,
                            exit_reason="signal",
                        )
                        append_trade(record)
                        pl     = (price - entry["buy_price"]) * entry["qty"]
                        pl_pct = (price - entry["buy_price"]) / entry["buy_price"] * 100
                        logger.info(
                            f"{symbol}: Closed by signal. "
                            f"${entry['buy_price']:.2f} → ${price:.2f} | P&L ${pl:.2f} ({pl_pct:.2f}%)"
                        )
                else:
                    logger.info(f"{symbol}: Bearish crossover — no position. MA9={ma_s_now:.2f} MA21={ma_l_now:.2f}")

            else:
                logger.info(f"{symbol}: Hold. MA9={ma_s_now:.2f} MA21={ma_l_now:.2f} Price=${price:.2f}")

        except Exception as e:
            logger.error(f"{symbol}: Unexpected error — {e}")

    logger.info("--- Strategy run complete ---")


# ── Flask dashboard ───────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    today_str = datetime.now(EST).strftime("%Y-%m-%d")

    # Open positions: merge in-memory state with Alpaca live prices
    with state_lock:
        trades_snapshot = dict(open_trades)
        spent           = daily_spend
        weights         = dict(symbol_weights)

    positions_out = []
    try:
        alpaca_positions = {p.symbol: p for p in trading_client.get_all_positions()}
    except Exception:
        alpaca_positions = {}

    for symbol, entry in trades_snapshot.items():
        ap      = alpaca_positions.get(symbol)
        cur_px  = float(ap.current_price)      if ap and ap.current_price      else entry["buy_price"]
        unr_pl  = float(ap.unrealized_pl)      if ap and ap.unrealized_pl      else 0.0
        positions_out.append({
            "symbol":        symbol,
            "qty":           entry["qty"],
            "buy_price":     entry["buy_price"],
            "buy_time":      entry["buy_time"],
            "current_price": round(cur_px, 4),
            "unrealized_pl": round(unr_pl, 4),
        })

    # Today's closed trades from CSV
    today_trades = []
    if Path(TRADE_LOG_FILE).exists():
        try:
            df = pd.read_csv(TRADE_LOG_FILE)
            if not df.empty:
                today_df = df[df["date"] == today_str]
                today_trades = today_df.to_dict(orient="records")
        except Exception:
            pass

    # Per-symbol stats from full history
    symbol_stats = []
    all_df = None
    if Path(TRADE_LOG_FILE).exists():
        try:
            all_df = pd.read_csv(TRADE_LOG_FILE)
            all_df["profit_loss"] = pd.to_numeric(all_df["profit_loss"], errors="coerce")
        except Exception:
            all_df = None

    for symbol in SYMBOLS:
        wins = losses = total = 0
        if all_df is not None and not all_df.empty:
            sdf    = all_df[all_df["symbol"] == symbol]
            total  = len(sdf)
            wins   = int((sdf["profit_loss"] > 0).sum())
            losses = total - wins
        symbol_stats.append({
            "symbol": symbol,
            "wins":   wins,
            "losses": losses,
            "total":  total,
            "weight": round(weights.get(symbol, 1 / len(SYMBOLS)), 4),
        })

    return jsonify({
        "open_positions":  positions_out,
        "today_trades":    today_trades,
        "symbol_stats":    symbol_stats,
        "daily_spend":     round(spent, 2),
        "budget_remaining": round(max(0, MAX_DAILY_SPEND - spent), 2),
        "max_daily_spend": MAX_DAILY_SPEND,
        "market_open":     is_market_open(),
        "as_of":           datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S EST"),
    })


def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ── Startup safety check ──────────────────────────────────────────────────────
def startup_safety_check() -> bool:
    """
    Verifies the account is safe to trade before the bot starts.
    Returns True (GO) or False (NO-GO) and always prints a clear status block.
    """
    mode_label = "PAPER TRADING" if PAPER_MODE else "*** LIVE TRADING — REAL MONEY ***"
    divider    = "=" * 60

    logger.info(divider)
    logger.info(f"  STARTUP SAFETY CHECK  |  Mode: {mode_label}")
    logger.info(divider)

    checks   = {}   # label -> (passed: bool, detail: str)
    go       = True

    # 1. API connectivity + fetch account
    try:
        account = trading_client.get_account()
    except Exception as e:
        logger.error(f"  [FAIL] Cannot connect to Alpaca: {e}")
        logger.info(divider)
        logger.info("  RESULT: NO-GO — fix API connectivity before starting.")
        logger.info(divider)
        return False

    # 2. Account status (alpaca-py returns an enum like AccountStatus.ACTIVE)
    raw_status = account.status
    status_str = (raw_status.value if hasattr(raw_status, "value") else str(raw_status)).upper()
    ok         = "ACTIVE" in status_str
    checks["Account status"] = (ok, status_str)
    if not ok:
        go = False

    # 3. Account not blocked
    acct_blocked = bool(account.account_blocked)
    checks["Account not blocked"] = (not acct_blocked, "blocked" if acct_blocked else "clear")

    # 4. Trading not blocked
    trade_blocked = bool(account.trading_blocked)
    checks["Trading not blocked"] = (not trade_blocked, "blocked" if trade_blocked else "clear")
    if acct_blocked or trade_blocked:
        go = False

    # 5. Minimum balance ($100)
    cash = float(account.cash)
    ok   = cash >= 100.0
    checks[f"Cash balance ≥ $100"] = (ok, f"${cash:,.2f}")
    if not ok:
        go = False

    # 6. Pattern Day Trader flag (warning only, doesn't block)
    pdt = bool(account.pattern_day_trader)
    checks["Pattern Day Trader flag"] = (
        not pdt,
        "PDT flagged — only 3 day-trades per 5 days unless account > $25k" if pdt else "not flagged",
    )

    # Print each check
    for label, (passed, detail) in checks.items():
        icon = "PASS" if passed else "FAIL"
        # PDT is just a warning
        if label.startswith("Pattern") and not passed:
            icon = "WARN"
        logger.info(f"  [{icon}] {label}: {detail}")

    # Overall result
    logger.info(divider)
    if go:
        logger.info(f"  RESULT: GO — bot will start in {mode_label} mode.")
    else:
        logger.info("  RESULT: NO-GO — resolve the issues above before starting.")
    logger.info(divider)

    return go


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_csv_exists()

    # Safety check — exits with code 1 on NO-GO
    if not startup_safety_check():
        sys.exit(1)

    logger.info("Trading bot starting up.")
    logger.info(f"Watching: {', '.join(SYMBOLS)}")
    logger.info(f"Strategy: MA{SHORT_MA}/MA{LONG_MA} | Max daily: ${MAX_DAILY_SPEND:.2f}")
    logger.info(f"Risk: {STOP_LOSS_PCT*100:.0f}% SL | {TAKE_PROFIT_PCT*100:.0f}% TP")
    logger.info("Initial budget allocations (equal weights):")
    for s in SYMBOLS:
        w = symbol_weights[s]
        logger.info(f"  {s}: {w*100:.1f}%  (${MAX_DAILY_SPEND * w:.2f})")

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Dashboard running on port 5000.")

    run_strategy()
    schedule.every(5).minutes.do(run_strategy)
    logger.info("Scheduler running — strategy every 5 min | analysis at 4:05 PM EST.")

    while True:
        schedule.run_pending()
        maybe_run_end_of_day_analysis()
        time_module.sleep(30)
