import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

api_key = os.environ.get("ALPACA_API_KEY")
secret_key = os.environ.get("ALPACA_SECRET_KEY")

client = TradingClient(api_key, secret_key, paper=True)

account = client.get_account()

print(f"Cash Balance:     ${float(account.cash):,.2f}")
print(f"Portfolio Value:  ${float(account.portfolio_value):,.2f}")
