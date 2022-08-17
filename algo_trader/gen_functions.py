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
TIMEFRAME = timedelta(weeks=300)
CALLS = 60
RATE_LIMIT = 60


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


def get_function_params(indicator, resolution='D', timeframe=TIMEFRAME, end=END):
    start = end - timeframe
    def start_stop(f): return {'_from': f(start), 'to': f(end)}

    if indicator == 'SOCIAL':
        return start_stop(get_day)
    params = start_stop(get_ts)
    params['resolution'] = resolution
    return params


def retrieve_data(ticker, indicator, retries=0, **kwargs):
    try:
        data = __base_function_switch(indicator)(ticker, **kwargs)
        return make_parquet(clean_response(data, indicator), ticker, indicator)
    except FinnhubAPIException as api:
        if api.status_code == 429 and retries < 3:
            retries += 1
            __api_wait(indicator, ticker)
            retrieve_data(indicator, ticker, retries, **kwargs)


def retrieve_indicator(tickers, indicator, **kwargs):
    _execute_ind = BoundedThreads(thread_name_prefix=indicator)
    for ticker in tickers:
        _execute_ind.addFuture(f"{ticker}_{indicator}", retrieve_data, ticker, indicator, **kwargs)
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


def column_map(df):
    return df.rename(columns={
        'c': 'close',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'v': 'volume',
    })


def clean_response(data, indicator):
    df = pd.DataFrame(data)
    df['timestamp'] = df.t.apply(lambda d: dt.fromtimestamp(d))
    if indicator == 'CANDLE':
        df = column_map(df)
    else:
        df = df[[c for c in df.columns if c not in 'cohlv']]
    return df.drop(columns=['s', 't'])


def make_parquet(dataframe, ticker, indicator):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param ticker: ticker from which data has been retrieved. sub-folder of output.
    :param indicator: stat(s) for ticker. base file name.
    :return: ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
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


def retrieve_all_indicators(tickers, indicators, **kwargs):
    meta_data = dict()
    for indicator in indicators:
        meta_data[indicator] = retrieve_indicator(tickers, indicator, **get_function_params(indicator, **kwargs))
    return meta_data
