import os
import logging
import time as time_module
from datetime import datetime, time, timedelta

import pytz
import schedule
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

SYMBOLS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
MAX_DAILY_SPEND = 80.0
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
SHORT_MA = 9
LONG_MA = 21
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EST = pytz.timezone("America/New_York")
TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET_KEY"]

trading_client = TradingClient(api_key, secret_key, paper=True)
data_client = StockHistoricalDataClient(api_key, secret_key)

daily_spend = 0.0
last_reset_date = None


def reset_daily_spend_if_needed():
    global daily_spend, last_reset_date
    today = datetime.now(EST).date()
    if last_reset_date != today:
        daily_spend = 0.0
        last_reset_date = today
        logger.info(f"Daily spend counter reset for {today}")


def is_market_open():
    now = datetime.now(EST)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() < MARKET_CLOSE


def get_bars(symbol):
    end = datetime.now(EST)
    start = end - timedelta(hours=6)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        limit=60,
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.loc[symbol]
    return df.reset_index(drop=True)


def get_position(symbol):
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


def place_bracket_order(symbol, qty, price):
    stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
    take_profit_price = round(price * (1 + TAKE_PROFIT_PCT), 2)
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class="bracket",
        stop_loss=StopLossRequest(stop_price=stop_price),
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
    )
    return trading_client.submit_order(order_request)


def run_strategy():
    global daily_spend

    reset_daily_spend_if_needed()

    if not is_market_open():
        logger.info("Market is closed. Skipping run.")
        return

    logger.info(
        f"--- Strategy run started | Daily spend: ${daily_spend:.2f} / ${MAX_DAILY_SPEND:.2f} ---"
    )

    for symbol in SYMBOLS:
        try:
            df = get_bars(symbol)

            if len(df) < LONG_MA + 1:
                logger.warning(f"{symbol}: Not enough bars ({len(df)}). Skipping.")
                continue

            df["ma_short"] = df["close"].rolling(SHORT_MA).mean()
            df["ma_long"] = df["close"].rolling(LONG_MA).mean()

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            ma_short_now = latest["ma_short"]
            ma_long_now = latest["ma_long"]
            ma_short_prev = prev["ma_short"]
            ma_long_prev = prev["ma_long"]
            price = float(latest["close"])

            if pd.isna(ma_short_now) or pd.isna(ma_long_now):
                logger.warning(f"{symbol}: MA values are NaN. Skipping.")
                continue

            crossed_above = (ma_short_now > ma_long_now) and (ma_short_prev <= ma_long_prev)
            crossed_below = (ma_short_now < ma_long_now) and (ma_short_prev >= ma_long_prev)

            position = get_position(symbol)

            if crossed_above:
                if position is not None:
                    logger.info(
                        f"{symbol}: Bullish crossover — already in position, no action. "
                        f"MA9={ma_short_now:.2f}, MA21={ma_long_now:.2f}"
                    )
                else:
                    budget_remaining = MAX_DAILY_SPEND - daily_spend
                    if budget_remaining <= 0:
                        logger.info(f"{symbol}: Bullish crossover — daily budget exhausted. Skipping buy.")
                        continue
                    qty = max(1, int(budget_remaining // price))
                    cost = qty * price
                    if cost > budget_remaining:
                        logger.info(
                            f"{symbol}: Bullish crossover — not enough budget "
                            f"(need ${cost:.2f}, have ${budget_remaining:.2f}). Skipping."
                        )
                        continue
                    logger.info(
                        f"{symbol}: Bullish crossover — buying {qty} share(s) at ~${price:.2f} "
                        f"(cost ~${cost:.2f}). MA9={ma_short_now:.2f}, MA21={ma_long_now:.2f}"
                    )
                    order = place_bracket_order(symbol, qty, price)
                    daily_spend += cost
                    logger.info(
                        f"{symbol}: Order placed (ID: {order.id}). "
                        f"Stop-loss: ${price*(1-STOP_LOSS_PCT):.2f}, "
                        f"Take-profit: ${price*(1+TAKE_PROFIT_PCT):.2f}. "
                        f"Daily spend now: ${daily_spend:.2f}"
                    )

            elif crossed_below:
                if position is not None:
                    logger.info(
                        f"{symbol}: Bearish crossover — closing position. "
                        f"MA9={ma_short_now:.2f}, MA21={ma_long_now:.2f}"
                    )
                    trading_client.close_position(symbol)
                    logger.info(f"{symbol}: Position closed.")
                else:
                    logger.info(
                        f"{symbol}: Bearish crossover — no open position, no action. "
                        f"MA9={ma_short_now:.2f}, MA21={ma_long_now:.2f}"
                    )

            else:
                logger.info(
                    f"{symbol}: No crossover. MA9={ma_short_now:.2f}, MA21={ma_long_now:.2f}, "
                    f"Price=${price:.2f}"
                )

        except Exception as e:
            logger.error(f"{symbol}: Unexpected error — {e}")

    logger.info("--- Strategy run complete ---")


if __name__ == "__main__":
    logger.info("Trading bot starting up.")
    logger.info(f"Watching: {', '.join(SYMBOLS)}")
    logger.info(f"Strategy: MA{SHORT_MA}/MA{LONG_MA} crossover | Max daily spend: ${MAX_DAILY_SPEND:.2f}")
    logger.info(f"Risk: {STOP_LOSS_PCT*100:.0f}% stop-loss | {TAKE_PROFIT_PCT*100:.0f}% take-profit")

    run_strategy()

    schedule.every(5).minutes.do(run_strategy)
    logger.info("Scheduler running — checking every 5 minutes.")

    while True:
        schedule.run_pending()
        time_module.sleep(30)
