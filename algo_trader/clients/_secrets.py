import os
from dotenv import load_dotenv
load_dotenv()

PAPER_TRADE = os.getenv('PAPER_TRADE', default=True)

"""
API for finnhub.io: https://finnhub.io/docs/api
"""
FINNHUB_KEY = os.getenv('FINNHUB_KEY')
FINNHUB_SANDBOX = os.getenv('FINNHUB_SANDBOX')
"""
API for Alpaca: https://alpaca.markets/docs/python-sdk/
"""
ALPACA_KEY = os.getenv('ALPACA_KEY')
ALPACA_SECRET = os.getenv('ALPACA_SECRET')
ALPACA_LIVE = os.getenv('PAPER') == 0
"""
API for Kaggle: https://www.kaggle.com/docs/api
"""
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')
