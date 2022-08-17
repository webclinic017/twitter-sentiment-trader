import time

from algo_trader import gen_functions as fgen

indicators = [
    'STOCH',
    'STOCHRSI',
    'RSI',
    'WILLR',
    'ADX',
    'ADXR',
    'APO',
    'MOM',
    'CMO',
    'AROON',
    'DX',
    'MINUSDI',
    'PLUSDI',
    'MINUSDM',
    'PLUSDM',
]


if __name__ == '__main__':
    from datetime import timedelta, datetime as dt
    from algo_trader import finnhubData
    tickers = finnhubData().company_peers('AAPL')
    test = fgen.pull_all_indicators(tickers, indicators, timeframe=timedelta(weeks=1200), resolution='D')
