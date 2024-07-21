

import loguru
from ta import momentum, volatility, trend
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.signal_service import add_signal
from core.utils import add_vertical_offset, generate_color
from pinepython.range_filter import RangeFilter
from .config import *


# Индикаторы MACD и RSI с новыми параметрами
def calculate_indicators(df: pd.DataFrame, conf_templates):
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

    # "Инверсия разрывов справедливой стоимости"
    df['ATR'] = volatility.average_true_range(df['high'], df['low'], df['close'], window=consts.get('atr_period'))
    df['Fair_Value'] = (df['price'] + df['price']) / 2
    df['Gap'] = df['Fair_Value'] - df['price'].shift(1)
    df['Signal_FVG'] = np.where(df['Gap'].abs() > df['ATR'] * consts.get('atr_mult'), np.sign(df['Gap']), 0)

    # Индикатор Stochastic RSI
    lengthRSI = consts.get('stoch_length_RSI')
    lengthStoch = consts.get('stoch_length_STOCH')
    smoothK = consts.get('stoch_smooth_K')
    smoothD = consts.get('stoch_smooth_D')
    upper_Band = consts.get('stoch_upper_band')
    lower_Band = consts.get('stoch_lower_band')

    rsi = momentum.RSIIndicator(df['close'], window=lengthRSI).rsi()
    stoch_rsi = momentum.StochasticOscillator(close=rsi, high=rsi, low=rsi, window=lengthStoch, smooth_window=smoothK)
    df['%K'] = stoch_rsi.stoch()
    df['%D'] = stoch_rsi.stoch_signal()

    df['Buy_SRsi'] = np.where((df['%K'] > df['%D']) & (df['%K'].shift(1) <= df['%D'].shift(1)), df['%D'], np.nan)
    df['Sell_SRsi'] = np.where((df['%K'] < df['%D']) & (df['%K'].shift(1) >= df['%D'].shift(1)), df['%D'], np.nan)

    df['Long_SRsi'] = np.where((df['Buy_SRsi'] <= lower_Band), True, False)
    df['Short_SRsi'] = np.where((df['Sell_SRsi'] >= upper_Band), True, False)

    # Генерация сигналов на основе пересечения индикаторов и фильтрации по RSI с учетом новых результатов
    df['MACD_FVG_BUY'] = 0
    df['MACD_FVG_SELL'] = 0

    if len(df) < 3:
        return df

    # Range Filter
    rf_mod = RangeFilter(df['close'], consts.get('rf_ema_window'), consts.get('rf_ema_mult'))
    rf_mod.calculate()

    df['RF_HIGH_BAND'] = rf_mod.hband
    df['RF_LOW_BAND'] = rf_mod.lband

    df['RF_BUY'] = rf_mod.longCondition
    df['RF_SELL'] = rf_mod.shortCondition

    # Conditions scope
    df['bollinger_condition'] = np.where(df['price'] < df['Bollinger_T_Low'], 1, np.where(df['price'] > df['Bollinger_T_High'], -1, 0))

    # Для macd_condition, используем сдвиг отдельно
    buy_macd = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
    sell_macd = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))
    df['macd_condition'] = np.where(buy_macd, 1, np.where(sell_macd, -1, 0))

    df['stoch_condition'] = np.where(pd.notna(df['Buy_SRsi']), 1, np.where(pd.notna(df['Sell_SRsi']), -1, 0))
    df['fvg_condition'] = np.where(df['Signal_FVG'] > 0, 1, np.where(df['Signal_FVG'] < 0, -1, 0))
    df['volatility_condition'] = np.where((consts.get('volatility_min') < df['Volatility_Deviation_Percent']) & (df['Volatility_Deviation_Percent'] < consts.get('volatility_max')), 1, -1)
    df['range_filter_condition'] = np.where(df['RF_BUY'], 1, np.where(df['RF_SELL'], -1, 0))

    for template in conf_templates:
        template_signal = np.zeros(len(df))

        # print('tmp', template_signal.head(10))
        for ind_num, ind in enumerate(template['indicators']):
            full_condition_buy = (df[f"{ind}_condition"] == 1)
            full_condition_sell = (df[f"{ind}_condition"] == -1)
            if ind_num > 0:
                for i_shift in range(1, consts.get('prev_bars_count_allowance')):
                    full_condition_buy |= (df[f"{ind}_condition"].shift(i_shift) == 1)
                    full_condition_sell |= (df[f"{ind}_condition"].shift(i_shift) == -1)
            template_signal += full_condition_buy.astype(int) - full_condition_sell.astype(int)
        # Применение нормализации сигнала
        df[f"{template['name']}_signal"] = template_signal.apply(lambda x: 1 if x > 0 and x == template_signal.max() else -1 if x < 0 and x == template_signal.min() else 0)

        # print(df.head(100))

    return df


def plot_signals(df: pd.DataFrame, symbol: str, fig: go.Figure, row: int, col: int):
    sl = row == 1 and col == 1
    if consts.get('candle_price_view'):

        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price', showlegend=sl,
                                     increasing=dict(line=dict(color='#0b9879', width=1), fillcolor='#0b9879'),
                                     decreasing=dict(line=dict(color='#da3354', width=1), fillcolor='#da3354')), row=row, col=col)

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
    if consts.get('bb_t_display_lines'):
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Bollinger_T_Low'], mode='lines', name=f'BB Trend Low', line=dict(color="rgba(255, 128, 128, 0.3)"), showlegend=sl), row=row, col=col)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Bollinger_T_High'], mode='lines', name=f'BB Trnd High', line=dict(color="rgba(100, 100, 255, 0.3)"), showlegend=sl), row=row, col=col)

    # Hband and lband
    if consts.get('rf_display_lines'):
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RF_HIGH_BAND'], mode='lines', name=f'Range Filter HIGH_BAND', line=dict(color="rgba(255, 255, 255, 0.7)"), showlegend=sl), row=row, col=col)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RF_LOW_BAND'], mode='lines', name=f'Range Filter LOW_BAND', line=dict(color="rgba(20, 20, 255, 0.4)"), showlegend=sl), row=row, col=col)

    # Signals
    for template in indicator_configs_template:

        buy_signals = df[df[f'{template['name']}_signal'] == 1]
        sell_signals = df[df[f'{template['name']}_signal'] == -1]

        # Добавляем вертикальный сдвиг к сигналам
        buy_signals_y = add_vertical_offset(buy_signals['low'].values, -settings['offset_p_macd_triangles'] / 100)
        sell_signals_y = add_vertical_offset(sell_signals['high'].values, settings['offset_p_macd_triangles'] / 100)

        # Рисуем сигналы
        fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals_y, mode='markers', marker=dict(symbol='triangle-up', color=template['up_arrow_color'], size=10), name=f'{template['name']} {consts.get(template['name'])} Buy Signal', showlegend=sl), row=row, col=col)
        fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals_y, mode='markers', marker=dict(symbol='triangle-down', color=template['down_arrow_color'], size=10), name=f'{template['name']} {consts.get(template['name'])} Sell Signal', showlegend=sl), row=row, col=col)

    # InvFVG
    # fig.add_trace(go.Scatter(x=df[df['InvFVG_BUY'].fillna(False) == 1]['timestamp'], y=df[df['InvFVG_BUY'].fillna(False) == 1]['price'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=20), name=f'FVG Buy Signal', showlegend=sl), row=row, col=col)
    # fig.add_trace(go.Scatter(x=df[df['InvFVG_SELL'].fillna(False) == 1]['timestamp'], y=df[df['InvFVG_SELL'].fillna(False) == 1]['price'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=20), name=f'FVG Buy Signal', showlegend=sl), row=row, col=col)

    # MACD
    if consts.get('use_macd'):
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

    # Stochastic RSI
    if consts.get('use_stoch'):
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['%K'], mode='lines', name=f'%K', line=dict(color="#f5514b"), showlegend=sl), row=row + 1, col=col)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['%D'], mode='lines', name=f'%D', line=dict(color="#5aa755"), showlegend=sl), row=row + 1, col=col)

        # Buy and Sell signals
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Buy_SRsi'], mode='markers', marker=dict(symbol='circle', color='#5aa755', size=10), name=f'Buy Signal', showlegend=sl), row=row + 1, col=col)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Sell_SRsi'], mode='markers', marker=dict(symbol='circle', color='#f5514b', size=10), name=f'Sell Signal', showlegend=sl), row=row + 1, col=col)


def process_symbol(data, symbol, fig, row, col):
    loguru.logger.debug(f"Processing symbol {symbol}")
    consts.set_symbol(symbol)
    df_symbol = data[data['symbol'] == symbol].copy().reset_index(drop=True)

    conf_templates = indicator_configs_template[:]
    for con in conf_templates:
        con['indicators'] = consts.get(con['name']).strip().split('+')

    df_symbol = calculate_indicators(df_symbol, conf_templates)

    plot_signals(df_symbol, symbol, fig, row, col)  # для графика цены
    last_n_records = df_symbol.tail(consts.get('max_bars_since_signal'))
    last_n_records = last_n_records.head(consts.get('max_bars_since_signal') - consts.get('min_bars_since_signal'))

    for template in conf_templates:
        s_name = template['name']
        for i, l_row in last_n_records.iterrows():
            if l_row[f'{s_name}_signal'] == 1 and (symbol, s_name, l_row['int_timestamp']) not in unique_signals:
                add_signal("buy", l_row['int_timestamp'], symbol, s_name, l_row['price'], df_symbol.iloc[-1]['price'])
            elif l_row[f'{s_name}_signal'] == -1 and (symbol, s_name, l_row['int_timestamp']) not in unique_signals:
                add_signal("sell", l_row['int_timestamp'], symbol, s_name, l_row['price'], df_symbol.iloc[-1]['price'])
    return fig