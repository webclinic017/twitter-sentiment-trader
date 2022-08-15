import pandas as pd
from json_normalize import json_normalize
from functools import partial
from itertools import cycle
from algo_trader.clients import alpacaData, finnhubData
from datetime import timedelta, datetime as dt
from distutils.dir_util import mkpath

# create folders
TRAIN_PATH = './trading_data/train'
TEST_PATH = './trading_data/test'

mkpath(TRAIN_PATH)
mkpath(TEST_PATH)

def get_ts(dval):
    return int(dval.timestamp())

def get_fin_method(method):
    with finnhubData() as fin:
        return getattr(fin, method)

def get_fin_part(method, *args, **kwargs): return partial(get_fin_method(method), *args, **kwargs)
def get_fin_part_range(method, *args, timeframe=timedelta(weeks=1), stop=dt.today(), **kwargs):
    return partial(get_fin_part(method), *args, _from=get_ts(stop-timeframe), to=get_ts(stop), resolution='D', **kwargs)

get_candles = get_fin_part_range('stock_candles')
get_sentiment = get_fin_part('stock_social_sentiment')

def get_technical(indicator):
    return partial(get_fin_part_range('technical_indicator', indicator=indicator))

def get_technical_many(indicators):
    return [get_technical(ind) for ind in indicators]

def get_all(indicators):
    func_lst = [get_candles, get_sentiment]
    return func_lst + get_technical_many(indicators)
