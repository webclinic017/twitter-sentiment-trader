from finnhub import Client
import alpaca_backtrader_api
from algo_trader.clients._secrets import ALPACA_KEY, ALPACA_SECRET, FINNHUB_KEY, FINNHUB_SANDBOX, PAPER_TRADE

store = alpaca_backtrader_api.AlpacaStore(
    key_id=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=PAPER_TRADE
)

# finnhub clients
def finnhubData():
    if PAPER_TRADE:
        return Client(api_key=FINNHUB_SANDBOX)
    return Client(api_key=FINNHUB_KEY)


def alpacaBroker(): return alpaca_backtrader_api.AlpacaBroker()
def alpacaData(): return alpaca_backtrader_api.AlpacaData()

