from alpaca.data.historical import StockHistoricalDataClient
from finnhub import Client
from .secrets import ALPACA_KEY, ALPACA_SECRET, FINNHUB_KEY, FINNHUB_SANDBOX

# keys required for stock historical data client
alpacaHistory = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
# finnhub clients
finnhubAPI = Client(api_key=FINNHUB_KEY)
finnhubSandbox = Client(api_key=FINNHUB_SANDBOX)
