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
        "limit": limit
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


# Генератор данных в необходимом формате
def generate_formatted_olhv(symbols):
    all_data = {}
    was_synced = False

    def download_and_format(symbol):
        last_cached = cached_prices.get(symbol)
        if last_cached:
            last_cached_time = last_cached[-1][0]
            current_time = int(datetime.now().timestamp())
            bars_diff = (current_time - last_cached_time) // timeframe_to_seconds(settings['timeframe'])
            limit = min(bars_diff, settings['last_n_bars'])
        else:
            limit = settings['last_n_bars']

        if not last_cached or bars_diff > 0:
            loguru.logger.info(f"Downloading symbol {symbol} with limit {limit}")
            ohlcv_data = download_binance_ohlcv(symbol, limit)
            try:
                formatted_data = [
                    (entry[0] // 1000, float(entry[1]), float(entry[2]), float(entry[3]), float(entry[4]), float(entry[5]))
                    for entry in ohlcv_data
                ]
                if last_cached:
                    formatted_data = last_cached + formatted_data
                cached_prices[symbol] = formatted_data
                return symbol, formatted_data
            except Exception as e:
                loguru.logger.error(f"Error on processing symbol {symbol} from binance: {e}, {traceback.format_exc()}")
                return symbol, []
        else:
            return symbol, last_cached

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
