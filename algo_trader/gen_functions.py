import os.path
import time
from functools import partial

import pandas as pd

from .clients import finnhubData
from .tools import *
from finnhub.exceptions import FinnhubAPIException
from datetime import timedelta, datetime as dt
from distutils.dir_util import mkpath
import pyarrow as pa
import pyarrow.parquet as pq


OUTDIR = './data/'
mkpath(OUTDIR)
END = dt.today()
TIMEFRAME = timedelta(weeks=1)
CALLS = 60
RATE_LIMIT = 60


def get_fin_method(method): return getattr(finnhubData(), method)
def get_fin_part(method, **kwargs): return partial(get_fin_method(method), **kwargs)


def get_technical(indicator, **kwargs): return get_fin_part(method='technical_indicator', indicator=indicator, **kwargs)


def get_candles(**kwargs): return get_fin_part('stock_candles', **kwargs)
def get_social(**kwargs): return get_fin_part('stock_social_sentiment', **kwargs)


def __base_function_switch(indicator, **kwargs):
    if indicator in ('CANDLE', 'SOCIAL'):
        return get_candles(**kwargs)
    elif indicator == 'SOCIAL':
        return get_social(**kwargs)
    return get_technical(indicator, **kwargs)


def get_day(dval): return str(dval)[:10]
def get_ts(dval): return int(dval.timestamp())


def get_function_params(indicator, resolution='D', timeframe=TIMEFRAME, end=END):
    start = end - timeframe
    def start_stop(f): return {'_from': f(start), 'to': f(end)}

    if indicator == 'SOCIAL':
        return start_stop(get_day)
    params = start_stop(get_ts)
    params['resolution'] = resolution
    return params


def pull_data(ticker, indicator,  **kwargs):
    try:
        func = __base_function_switch(indicator)
        data = func(symbol=ticker, **kwargs)
        df = clean_response(data)
        return make_parquet(df, ticker, indicator)
    except FinnhubAPIException as api:
        if api.status_code == 429:
            __api_wait(indicator, ticker)
            return pull_data(indicator, ticker, **kwargs)


def pull_indicator(tickers, indicator, **kwargs):
    _execute_ind = BoundedThreads(thread_name_prefix=indicator)
    for ticker in tickers:
        _execute_ind.addFuture(f"{ticker}_{indicator}", pull_data, ticker, indicator, **kwargs)
    return _execute_ind.check_futures()


def __api_wait(indicator, ticker):
    print(
        str_format(
            f"""
            {stars}
            Finnhub <Response 429>: API limit reached! Retrying the following parameters in {RATE_LIMIT} seconds:
            {dashes}
            INDICATOR: {indicator}
            TICKER: {ticker}
            {stars}
            """
        ),
        end='\n'
    )
    time.sleep(RATE_LIMIT)


def clean_response(data):
    df = pd.DataFrame(data)
    df['date'] = df.t.apply(lambda d: dt.fromtimestamp(d).date())
    df.set_index(keys='date', inplace=True)
    return df[[c for c in df.columns if c not in 'ochlvst']]


def make_parquet(dataframe, ticker, indicator):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param ticker: ticker from which data has been pulled. sub-folder of output.
    :param indicator: stat(s) for ticker. base file name.
    :return: ParquetWriter object. This can be passed in the subsequent method calls to append DataFrame
        in the pyarrow Table
    """

    table = pa.Table.from_pandas(df=dataframe)
    tickdir = os.path.join(OUTDIR, ticker.replace('.', '-'))
    mkpath(tickdir)
    fpath = os.path.join(tickdir, f"{indicator}.parquet")
    with pq.ParquetWriter(fpath, table.schema) as writer:
        writer.write_table(table=table)
    return {
        'meta data': pq.read_metadata(fpath),
        'description': dataframe.describe()
    }


def pull_all_indicators(tickers, indicators, **kwargs):
    meta_data = dict()
    for indicator in indicators:
        meta_data[indicator] = pull_indicator(tickers, indicator, **get_function_params(indicator, **kwargs))
    return meta_data


def get_ticker_df(ticker):
    ticker = ticker.upper()
    tickerdir = os.path.join(OUTDIR, ticker)
    df = pd.DataFrame()
    for file in os.listdir(tickerdir):
        fname = os.path.join(tickerdir, file)
        df = pd.concat([df, pd.read_parquet(fname)], axis=1)
    return df.T.drop_duplicates().T
