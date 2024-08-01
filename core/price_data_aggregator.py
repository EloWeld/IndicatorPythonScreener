from datetime import datetime
import traceback
import loguru
import pandas as pd
import requests
import concurrent.futures

from core.config import *
from core.utils import *

# Функция для получения данных с биржи Binance


def download_binance_ohlcv(symbol, limit):
    data = []
    if settings.get('is_using_spot', False):
        url = "https://api.binance.com/api/v3/klines"
    else:
        url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": settings['timeframe'],
        "limit": limit + 5
    }
    response = None
    try:
        if settings.get('socks5_proxy', None):
            response = requests.get(url, params=params, proxies={'http': settings['socks5_proxy'], 'https': settings['socks5_proxy']})
        else:
            response = requests.get(url, params=params)
        data = response.json()
    except Exception as e:
        if response is not None:
            loguru.logger.error(f"Error binance! {response.text}, {e}")
        else:
            loguru.logger.error(f"Error on request to binance! {e}")
    return data


def download_biybit_ohlv(symbol, limit):
    data = []
    url = "https://api.bybit.com/v5/market/kline"
    minutes = timeframe_to_seconds(settings['timeframe']) // 60
    params = {
        "category": "spot" if settings.get('is_using_spot', False) else "linear",
        "symbol": symbol,
        "interval": "W" if minutes == 60*24*7 else "M" if minutes == 60*24*30 else "D" if minutes == 60*24 else minutes,
        "limit": limit + 5  # изменено
    }
    response = None
    try:
        if settings.get('socks5_proxy', None):
            response = requests.get(url, params=params, proxies={'http': settings['socks5_proxy'], 'https': settings['socks5_proxy']})
        else:
            response = requests.get(url, params=params)
        data = response.json()
        data = [x for x in data['result']['list']]
    except Exception as e:
        if response is not None:
            loguru.logger.error(f"Error bybit! {response.text}, {e}, {params}")
        else:
            loguru.logger.error(f"Error on request to bybit! {e}")
    return data


# Генератор данных в необходимом формате
def generate_formatted_olhv(symbols):
    all_data = {}
    was_synced = False

    def download_and_format(symbol):
        # last_cached = cached_prices.get(symbol)
        # if last_cached:
        #     last_cached_time = last_cached[-1][0]
        #     current_time = int(datetime.now().timestamp())
        #     bars_diff = (current_time - last_cached_time) // timeframe_to_seconds(settings['timeframe'])
        #     limit = max(min(bars_diff + 5, settings['last_n_bars'] + 5), 6)
        # else:
        #     limit = settings['last_n_bars'] + 5
        limit = settings['last_n_bars']

        loguru.logger.info(f"Downloading symbol {symbol} with limit {limit}")
        if settings['exchange'] == "bybit":
            ohlcv_data = download_biybit_ohlv(symbol, limit)
        elif settings['exchange'] == "binance":
            ohlcv_data = download_binance_ohlcv(symbol, limit)
        else:
            ohlcv_data = []

        try:
            if isinstance(ohlcv_data, dict) and ohlcv_data.get("retCode", None):
                loguru.logger.error(f"Ret code error on processing symbol {symbol} from bybit: {ohlcv_data}")
                return symbol, []
            formatted_data = [
                (int(entry[0]) // 1000, float(entry[1]), float(entry[2]), float(entry[3]), float(entry[4]), float(entry[5]))
                for entry in ohlcv_data
            ]
            # if last_cached:
            #     # Удалить последние 5 баров, чтобы перезаписать их
            #     last_cached = last_cached[:-5]
            #     formatted_data = formatted_data
            cached_prices[symbol] = formatted_data
            return symbol, formatted_data
        except Exception as e:
            loguru.logger.error(f"Error on processing symbol {symbol} from exchange: {e}, {traceback.format_exc()}")
            return symbol, []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(download_and_format, symbols))

    for symbol, data in results:
        all_data[symbol] = data[-settings['last_n_bars']:]
        if data:
            was_synced = True

    if was_synced:
        settings['last_sync_time'] = datetime.now().isoformat()
        save_cached(settings, cached_prices)

    return all_data


def process_data(data):
    price_data = [
        {"symbol": symbol, "timestamp": datetime.fromtimestamp(ts), "open": open, "high": high, "low": low, "price": close, "close": close, "volume": volume}
        for symbol, entries in data.items()
        for ts, open, high, low, close, volume in entries
    ]

    df = pd.DataFrame(price_data)
    return df
