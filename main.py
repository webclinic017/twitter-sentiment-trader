from algo_trader import gen_functions as fgen

indicators = [
    'SMA',
    'EMA',
    'WMA',
    'DEMA',
    'TEMA',
    'TRIMA',
    'KAMA',
    'MACD',
    'MACDEXT',
    'STOCH',
    'STOCHF',
    'RSI',
    'STOCHRSI',
    'WILLR',
    'ADX',
    'ADXR',
    'APO',
    'PPO',
    'MOM',
    'BOP',
    'CCI',
    'CMO',
    'ROC',
    'ROCR',
    'AROON',
    'AROONOSC',
    'MFI',
    'TRIX',
    'ULTOSC',
    'DX',
    'MINUSDI',
    'PLUSDI',
    'MINUSDM',
    'PLUSDM',
    'BBANDS',
    'MIDPOINT',
    'MIDPRICE',
    'SAR',
    'TRANGE',
    'ATR',
    'NATR',
    'AD',
    'ADOSC',
    'OBV',
    'CANDLE',
]


if __name__ == '__main__':
    from datetime import timedelta
    from algo_trader import finnhubData
    tickers = finnhubData().company_peers('AAPL')
    test = fgen.retrieve_all(tickers, indicators, timeframe=timedelta(weeks=500))
