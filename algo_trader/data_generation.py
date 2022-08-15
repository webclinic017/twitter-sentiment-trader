from functools import partial
from .clients import finnhubClient
from datetime import timedelta, datetime as dt
from distutils.dir_util import mkpath

# create folders
TRAIN_PATH = './trading_data/train'
TEST_PATH = './trading_data/test'

mkpath(TRAIN_PATH)
mkpath(TEST_PATH)


END = dt.today()
TIMEFRAME = timedelta(weeks=300)
START = END-TIMEFRAME

def get_fin_method(method):
    return getattr(finnhubClient, method)

def get_fin_part(method, *args, **kwargs): return partial(get_fin_method(method), *args, **kwargs)

get_candles = get_fin_part('stock_candles')
get_sentiment = get_fin_part('stock_social_sentiment')

def get_technical(indicator):return get_fin_part('technical_indicator', indicator=indicator)

def get_technical_many(indicators):
    for ind in indicators:
        yield get_technical(ind)

def get_all_functions(indicators):
    funcs = dict(zip(indicators, list(get_technical_many(indicators))))
    funcs['CNDL'] = get_candles
    funcs['SOCIAL'] = get_sentiment
    return funcs



