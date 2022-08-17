import time
from functools import partial

import pandas as pd

from ratelimit import limits, sleep_and_retry

CALLS = 60
RATE_LIMIT = 60

from .clients import finnhubData
from .tools import *
from finnhub.exceptions import FinnhubAPIException
from datetime import timedelta, datetime as dt
from distutils.dir_util import mkpath

# create folders
TRAIN_PATH = './trading_data/train'
TEST_PATH = './trading_data/test'
RAW_PATH = './trading_data/raw'

mkpath(TRAIN_PATH)
mkpath(TEST_PATH)

END = dt.today()
TIMEFRAME = timedelta(weeks=300)
START = END - TIMEFRAME


def get_fin_method(method): return getattr(finnhubData(), method)
def get_fin_part(method, *args, **kwargs): return partial(get_fin_method(method), *args, **kwargs)
def get_technical(indicator): return get_fin_part('technical_indicator', indicator=indicator)


get_candles = get_fin_part('stock_candles')
get_social = get_fin_part('stock_social_sentiment')


def __base_function_switch(indicator):
    if indicator == 'CANDLE':
        return get_candles
    elif indicator == 'SOCIAL':
        return get_social
    return get_technical(indicator)


def get_day(dval): return str(dval)[:10]
def get_ts(dval): return int(dval.timestamp())


def get_function_timeframe(indicator, resolution='D', timeframe=timedelta(weeks=1), end=dt.today()):
    start = end - timeframe
    func = __base_function_switch(indicator)

    def get_params(f): return {'resolution': resolution, '_from': f(start), 'to': f(end)}

    if indicator == 'SOCIAL':
        return partial(func, **get_params(get_day))
    return partial(func, **get_params(get_ts))


def retrieve_data(indicator, ticker, **kwargs):
    try:
        return {
            'status': 200,
            'data': get_function_timeframe(indicator, **kwargs)(ticker)
        }
    except FinnhubAPIException as api:
        return {
            'status': api.status_code,
            'data': dict()
        }


def retrieve_indicator(tickers, indicator, **kwargs):
    _execute = BoundedThreads(thread_name_prefix=indicator)
    print(indicator, tickers)
    for ticker in tickers:
        _execute.addFuture(ticker, retrieve_data, indicator, ticker, **kwargs)
    return pd.DataFrame(_execute.check_futures()).T


def retrieve_all(tickers, indicators, **kwargs):
    responses = dict()
    for indicator in indicators:
        response = retrieve_indicator(tickers, indicator, **kwargs)
        passed = response[response.status == 200]
        failed = response[response.status == 429]
        if not failed.empty:
            rerun = failed.index.to_list()
            print(str_format(f"""
            {stars}
            API limit reached! Retrying the following parameters in {RATE_LIMIT} seconds
            {dashes}
            INDICATOR: {indicator}
            TICKERS: {rerun}
            {stars}
            """))
            time.sleep(RATE_LIMIT)
            passed = pd.concat([passed, retrieve_indicator(rerun, indicator)])
        responses[indicator] = passed.data.to_dict()
    return responses
