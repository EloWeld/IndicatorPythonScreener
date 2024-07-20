import hashlib
from itertools import chain
import json
import os
import platform
import random
import time
import traceback
import webbrowser
import loguru
import pandas as pd
import numpy as np
import pygame
import requests
from ta import volatility, trend, momentum
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
from utils import *
import concurrent.futures

from pinepython.range_filter import RangeFilter
from config import *


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
        all_data[symbol] = data
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


def add_signal(side: str, timestamp, symbol, name):
    if side == "buy":
        sound_up.play()
    else:
        sound_down.play()
    unique_signals.add((symbol, name, timestamp))
    save_signals()
    loguru.logger.success(f"New signal! {symbol}; {side}; {name}")

    # Notify in telegram bot
    bot_token = consts.get('telegram_bot_token')
    user_ids = consts.get('telegram_user_ids')

    message = f"{'🟢' if side == 'buy' else '🔴'} Signal: <code>{side}</code>\nTime: <code>{datetime.fromtimestamp(timestamp / 1000).strftime('%m.%d %H:%M')}</code>\nSymbol: <code>{symbol}</code>\nDescription: <code>{name}</code>"
    telegram_api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    for user_id in user_ids:
        try:
            response = requests.post(
                telegram_api_url,
                data={'chat_id': user_id, 'text': message, 'parse_mode': 'html'},

            )
            if response.status_code != 200:
                loguru.logger.error(f"Failed to send message to {user_id}. Status code: {response.status_code}, response: {response.text}")
        except Exception as e:
            loguru.logger.error(f"Error while sending Telegram message! Error: {e}; Traceback: {traceback.format_exc()}")

    # Send webhook
    webhook_data_str = "?"
    category = 'macd' if 'macd' in name.lower() else 'rf'
    try:
        webhook_data_str = json.dumps(consts.get(f'{category}_{side}_webhook_data'))
        webhook_data_str = webhook_data_str.replace('{{symbol}}', symbol).replace('{{side}}', side).replace('{{desc}}', name)
        webhook_data = json.loads(webhook_data_str)

        resp = requests.post(consts.get(f'{category}_{side}_webhook_url'), json=webhook_data)
        if resp.status_code != 200:
            loguru.logger.error(f"Not success status code while sending webhook! Status code: {resp.status_code}, data: {resp.text}")
    except Exception as e:
        loguru.logger.error(f"Error while sending webhook! sending_data: {webhook_data_str}; Error: {e}; Traceback: {traceback.format_exc()}")


# Индикаторы MACD и RSI с новыми параметрами
def calculate_indicators(df):
    df['MACD'] = trend.ema_indicator(df['price'], window=12) - trend.ema_indicator(df['price'], window=26)
    df['Signal_Line'] = trend.ema_indicator(df['MACD'], window=9)
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['RSI'] = momentum.rsi(df['price'], window=20)
    df['Bollinger_V_High'] = volatility.bollinger_hband(df['price'], window=consts.get('bb_v_window'), window_dev=consts.get('bb_v_deviation'))
    df['Bollinger_V_Low'] = volatility.bollinger_lband(df['price'], window=consts.get('bb_v_window'), window_dev=consts.get('bb_v_deviation'))
    df['Bollinger_T_High'] = volatility.bollinger_hband(df['price'], window=consts.get('bb_t_window'), window_dev=consts.get('bb_t_deviation'))
    df['Bollinger_T_Low'] = volatility.bollinger_lband(df['price'], window=consts.get('bb_t_window'), window_dev=consts.get('bb_t_deviation'))
    df['Bollinger_Volatility'] = df['Bollinger_V_High'] - df['Bollinger_V_Low']
    # Расчет среднего значения волатильности
    df['Avg_Bollinger_Volatility'] = df['Bollinger_Volatility'].rolling(window=consts.get('bb_dev_ma_window')).mean()

    # Расчет отклонения текущей волатильности от среднего значения
    df['Volatility_Deviation'] = df['Bollinger_Volatility'] - df['Avg_Bollinger_Volatility']
    df['Volatility_Deviation_Percent'] = 100 * (df['Bollinger_Volatility'] - df['Avg_Bollinger_Volatility']) / df['Avg_Bollinger_Volatility']

    return df


# Новая функция для индикатора "Инверсия разрывов справедливой стоимости"
def fair_value_gap_inversion(df, atr_period=200, atr_multi=0.3):
    df['ATR'] = volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_period)
    df['Fair_Value'] = (df['price'] + df['price']) / 2
    df['Gap'] = df['Fair_Value'] - df['price'].shift(1)
    df['Signal_FVG'] = np.where(df['Gap'].abs() > df['ATR'] * atr_multi, np.sign(df['Gap']), 0)

    return df


# Генерация сигналов на основе пересечения индикаторов и фильтрации по RSI с учетом новых результатов
def ind_macd_fvg_bb(df):
    df['MACD_FVG_BUY'] = 0
    df['MACD_FVG_SELL'] = 0

    if len(df) < 3:
        return df

    condition1 = (df['price'] < df['Bollinger_T_Low']) | (df['price'] > df['Bollinger_T_High'])
    # Variant 1
    gap_signal = df['Signal_FVG']
    condition2 = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)) & (gap_signal > 0)
    condition3 = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)) & (gap_signal < 0)
    # Variant 2
    # a = FVGIndicator(df, 5)
    # df = a.run()
    # gap_signal = df['InvFVG_BUY'] - df['InvFVG_SELL']
    # condition2 = (((df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))) | ((df['MACD'].shift(1) > df['Signal_Line'].shift(1)) & (df['MACD'].shift(2) <= df['Signal_Line'].shift(2)))) & ((gap_signal < 0) | (gap_signal.shift(1) < 0) | (gap_signal.shift(2) < 0))
    # condition3 = (((df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))) | ((df['MACD'].shift(1) < df['Signal_Line'].shift(1)) & (df['MACD'].shift(2) >= df['Signal_Line'].shift(2)))) & ((gap_signal > 0) | (gap_signal.shift(1) > 0) | (gap_signal.shift(2) > 0))

    cond_v = (consts.get('volatility_min') < df['Volatility_Deviation_Percent']) & (df['Volatility_Deviation_Percent'] < consts.get('volatility_max'))

    df.loc[condition2 & condition1 & cond_v, 'MACD_FVG_BUY'] = 1
    df.loc[condition3 & condition1 & cond_v, 'MACD_FVG_SELL'] = -1

    return df


# Range Filter
def ind_range_filter(df: pd.DataFrame, period=100, multiplier=3.0):
    df_copy = df.reset_index(drop=True)
    rf_mod = RangeFilter(df_copy['close'], period, multiplier)
    rf_mod.calculate()

    cond_v = (consts.get('volatility_min') < df['Volatility_Deviation_Percent']) & (df['Volatility_Deviation_Percent'] < consts.get('volatility_max'))
    df['RF_BUY'] = rf_mod.longCondition
    df.loc[~cond_v, 'RF_BUY'] = False
    df['RF_SELL'] = rf_mod.shortCondition
    df.loc[~cond_v, 'RF_SELL'] = False
    df['RF_HIGH_BAND'] = rf_mod.hband
    df['RF_LOW_BAND'] = rf_mod.lband

    return df


def plot_signals(df: pd.DataFrame, symbol: str, fig: Figure, row: int, col: int):
    sl = row == 1 and col == 1
    if consts.get('candle_price_view'):

        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price', showlegend=sl), row=row, col=col)

    else:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=f'Price', line=dict(color=generate_color(symbol), width=2), showlegend=sl), row=row, col=col)

    # BB volatility
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Volatility_Deviation'] + df['price'].min() - df['Volatility_Deviation'].max(), mode='lines', name=f'BB Volatility', line=dict(color="rgba(30, 200, 30, 0.7)"), yaxis='y2', showlegend=sl), row=row, col=col)
    df['volatility_min'] = consts.get('volatility_min') / 100
    df['volatility_min'] = (df['volatility_min'] * df['Avg_Bollinger_Volatility'])
    df['volatility_max'] = consts.get('volatility_max') / 100
    df['volatility_max'] = (df['volatility_max'] * df['Avg_Bollinger_Volatility'])
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['volatility_min'] + df['price'].min() - df['Volatility_Deviation'].max(), mode='lines', name=f'BB Volatility', line=dict(color="rgba(200, 20, 30, 0.7)"), yaxis='y2', showlegend=sl), row=row, col=col)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['volatility_max'] + df['price'].min() - df['Volatility_Deviation'].max(), mode='lines', name=f'BB Volatility', line=dict(color="rgba(200, 20, 30, 0.7)"), yaxis='y2', showlegend=sl), row=row, col=col)

    # BB trend
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Bollinger_T_Low'], mode='lines', name=f'BB Trend Low', line=dict(color="rgba(255, 128, 128, 0.3)"), showlegend=sl), row=row, col=col)
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Bollinger_T_High'], mode='lines', name=f'BB Trnd High', line=dict(color="rgba(100, 100, 255, 0.3)"), showlegend=sl), row=row, col=col)

    # Hband and lband
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RF_HIGH_BAND'], mode='lines', name=f'Range Filter HIGH_BAND', line=dict(color="rgba(255, 255, 255, 0.7)"), showlegend=sl), row=row, col=col)
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RF_LOW_BAND'], mode='lines', name=f'Range Filter LOW_BAND', line=dict(color="rgba(20, 20, 255, 0.4)"), showlegend=sl), row=row, col=col)

    # Signals
    buy_signals = df[df['MACD_FVG_BUY'] == 1]
    sell_signals = df[df['MACD_FVG_SELL'] == -1]
    rf_buy_signals = df[df['RF_BUY'].fillna(False)]
    rf_sell_signals = df[df['RF_SELL'].fillna(False)]

    # Добавляем вертикальный сдвиг к сигналам
    buy_signals_y = add_vertical_offset(buy_signals['low'].values, -settings['offset_p_macd_triangles'] / 100)
    sell_signals_y = add_vertical_offset(sell_signals['high'].values, settings['offset_p_macd_triangles'] / 100)
    rf_buy_signals_y = add_vertical_offset(rf_buy_signals['low'].values, -settings['offset_p_rf_triangles'] / 100)
    rf_sell_signals_y = add_vertical_offset(rf_sell_signals['high'].values, settings['offset_p_rf_triangles'] / 100)

    # Рисуем сигналы
    fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals_y, mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name=f'MACD+FVG+BB Buy Signal', showlegend=sl), row=row, col=col)
    fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals_y, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name=f'MACD+FVG+BB Sell Signal', showlegend=sl), row=row, col=col)
    fig.add_trace(go.Scatter(x=rf_buy_signals['timestamp'], y=rf_buy_signals_y, mode='markers', marker=dict(symbol='triangle-up', color='orange', size=10), name=f'RF Buy Signal', showlegend=sl), row=row, col=col)
    fig.add_trace(go.Scatter(x=rf_sell_signals['timestamp'], y=rf_sell_signals_y, mode='markers', marker=dict(symbol='triangle-down', color='blue', size=10), name=f'RF Sell Signal', showlegend=sl), row=row, col=col)

    # InvFVG
    # fig.add_trace(go.Scatter(x=df[df['InvFVG_BUY'].fillna(False) == 1]['timestamp'], y=df[df['InvFVG_BUY'].fillna(False) == 1]['price'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=20), name=f'FVG Buy Signal', showlegend=sl), row=row, col=col)
    # fig.add_trace(go.Scatter(x=df[df['InvFVG_SELL'].fillna(False) == 1]['timestamp'], y=df[df['InvFVG_SELL'].fillna(False) == 1]['price'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=20), name=f'FVG Buy Signal', showlegend=sl), row=row, col=col)

    # MACD
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD'], mode='lines', name=f'MACD', line=dict(color="green"), showlegend=sl),
                  row=row + 1, col=col)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Signal_Line'], mode='lines', name=f'MACD Signal Line', line=dict(color="red"), showlegend=sl),
                  row=row + 1, col=col)
    # Вычисляем разницу значений MACD_Hist
    macd_diff = np.diff(df['MACD_Hist'], prepend=np.nan)

    # Определяем цвета на основе условий
    colors = np.where((df['MACD_Hist'] < 0) & (macd_diff < 0), 'darkred',
                      np.where((df['MACD_Hist'] < 0) & (macd_diff >= 0), 'lightcoral',
                               np.where((df['MACD_Hist'] >= 0) & (macd_diff > 0), 'green',
                                        np.where((df['MACD_Hist'] >= 0) & (macd_diff <= 0), 'lightgreen', 'black'))))

    # Создаем гистограмму с использованием массива цветов
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['MACD_Hist'], name=f'MACD Histogram', marker_color=colors, showlegend=sl
                         ), row=row + 1, col=col)


# Инициализация микшера Pygame
pygame.mixer.init()

sound_up = pygame.mixer.Sound(resource_path('sound_up.mp3'))
sound_down = pygame.mixer.Sound(resource_path('sound_down.mp3'))


def main():

    is_browser_opened = False
    symbols = settings['symbols']
    html_file = resource_path('trading_signals.html')

    # Clear html file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write("")

    while True:
        data = generate_formatted_olhv(symbols)
        df = process_data(data)

        # Обновляем создание subplots для 2 колонок
        cols = 2
        rows = (len(symbols) * 2 + 2) // 2  # Расчет количества строк для двух столбцов, каждая монета занимает 2 строки (Price и MACD)
        subplot_titles = []
        for i in range(0, len(symbols), 2):
            if i+1 < len(symbols):
                subplot_titles.extend([f"{symbols[i]} Price", f"{symbols[i+1]} Price"])
                subplot_titles.extend([f"{symbols[i]} MACD", f"{symbols[i+1]} MACD"])
            else:
                subplot_titles.extend([f"{symbols[i]} Price", "-"])
                subplot_titles.extend([f"{symbols[i]} MACD", "-"])
        fig = make_subplots(
            rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.1 / rows, horizontal_spacing=0.02,
            subplot_titles=subplot_titles,
            row_heights=[0.4 if i % 2 else 1 for i in range(rows)], column_widths=[0.5, 0.5]
        )

        df['int_timestamp'] = df['timestamp'].apply(lambda x: int(x.timestamp()))

        row = 1
        col = 1
        for index, symbol in enumerate(symbols):
            loguru.logger.debug(f"Processing symbol {symbol}")
            consts.set_symbol(symbol)
            df_symbol = df[df['symbol'] == symbol].copy().reset_index(drop=True)
            df_symbol = calculate_indicators(df_symbol)
            df_symbol = fair_value_gap_inversion(df_symbol, atr_period=consts.get('atr_period'), atr_multi=consts.get('atr_mult'))
            df_symbol = ind_range_filter(df_symbol, consts.get('rf_ema_window'), consts.get('rf_ema_mult'))
            df_symbol = ind_macd_fvg_bb(df_symbol)

            if index % 2 == 0 and index != 0:
                row += 2
            if index % 2 == 0:
                col = 1
            else:
                col = 2

            # Отображаем графики для каждой монеты
            plot_signals(df_symbol, symbol, fig, row, col)  # для графика цены
            last_n_records = df_symbol.tail(consts.get('max_bars_since_signal'))
            last_n_records = last_n_records.head(consts.get('max_bars_since_signal') - consts.get('min_bars_since_signal'))

            for i, l_row in last_n_records.iterrows():
                if l_row['RF_BUY'] == 1 and (symbol, 'RF_BUY', l_row['int_timestamp']) not in unique_signals:
                    add_signal("buy", l_row['int_timestamp'], symbol, 'RF_BUY')
                if l_row['MACD_FVG_BUY'] == 1 and (symbol, 'MACD_FVG_BUY', l_row['int_timestamp']) not in unique_signals:
                    add_signal("buy", l_row['int_timestamp'], symbol, 'MACD_FVG_BUY')
                if l_row['RF_SELL'] == 1 and (symbol, 'RF_SELL', l_row['int_timestamp']) not in unique_signals:
                    add_signal("sell", l_row['int_timestamp'], symbol, 'RF_SELL')
                if l_row['MACD_FVG_SELL'] == -1 and (symbol, 'MACD_FVG_SELL', l_row['int_timestamp']) not in unique_signals:
                    add_signal("sell", l_row['int_timestamp'], symbol, 'MACD_FVG_SELL')
                # while pygame.mixer.get_busy():
                #     pygame.time.delay(10)

        fig.update_layout(title_text='Trading Signals for 20 Cryptocurrencies',
                          height=settings['height']*rows,
                          width=settings['width']*cols)

        # Отключение range slider для всех осей
        for axis_num in range(1, rows * cols + 1):
            fig.update_layout(**{f'xaxis{axis_num}_rangeslider_visible': False})

        fig.show()
        fig.write_html(html_file)
        with open(html_file, 'r+', encoding='utf-8') as f:
            content = f.read()
            f.seek(0)
            f.write(content.replace('</head>', f'<meta http-equiv="refresh" content="{settings['update_rate_minutes']*60}"></head>'))
        if not is_browser_opened:
            open_html_file(html_file)
            is_browser_opened = True

        time.sleep(settings['update_rate_minutes'] * 60)  # Обновление каждые n минуты


if __name__ == "__main__":
    main()
