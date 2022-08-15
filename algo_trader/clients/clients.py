import finnhub
import alpaca_backtrader_api
from algo_trader.clients._secrets import ALPACA_KEY, ALPACA_SECRET, FINNHUB_KEY, FINNHUB_SANDBOX, PAPER_TRADE

store = alpaca_backtrader_api.AlpacaStore(
    key_id=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=PAPER_TRADE
)

# finnhub client
class finnhubClient(finnhub.Client):
    def __init__(self):
        if PAPER_TRADE:
            api_key = FINNHUB_SANDBOX
        else:
            api_key = FINNHUB_KEY
        super(finnhubClient, self).__init__(api_key)


def alpacaBroker(): return alpaca_backtrader_api.AlpacaBroker()
def alpacaData(): return alpaca_backtrader_api.AlpacaData()
