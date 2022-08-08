from alpaca.trading.client import TradingClient
from .secrets import ALPACA_KEY, ALPACA_SECRET

# paper=True enables paper trading
traderAPI = TradingClient(ALPACA_KEY, ALPACA_SECRET)
paperAPI = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
