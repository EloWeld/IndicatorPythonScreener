from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output

import loguru
import pandas as pd
import numpy as np
from ta import volatility, trend, momentum
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
from core.price_data_aggregator import generate_formatted_olhv, process_data
from core.signal_service import add_signal
from core.utils import *

from pinepython.range_filter import RangeFilter
from core.config import *


indicator_configs_template = [
    {
        "name": "green_red",
        "up_arrow_color": "green",
        "down_arrow_color": "red",
        "indicators": []
    },
    {
        "name": "yellow_blue",
        "up_arrow_color": "yellow",
        "down_arrow_color": "blue",
        "indicators": []
    }
]


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
    df['bollinger_condition'] = df.apply(lambda row: 1 if row['price'] < row['Bollinger_T_Low'] else -1 if row['price'] > row['Bollinger_T_High'] else 0, axis=1)

    # Для macd_condition, используем сдвиг отдельно
    df['macd_condition'] = 0
    buy_macd = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
    sell_macd = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))
    df.loc[buy_macd, 'macd_condition'] = 1
    df.loc[sell_macd, 'macd_condition'] = -1

    df['stoch_condition'] = df.apply(lambda row: 1 if pd.notna(row['Buy_SRsi']) else -1 if pd.notna(row['Sell_SRsi']) else 0, axis=1)
    df['fvg_condition'] = df.apply(lambda row: 1 if row['Signal_FVG'] > 0 else -1 if row['Signal_FVG'] < 0 else 0, axis=1)
    df['volatility_condition'] = df.apply(lambda row: 1 if consts.get('volatility_min') < row['Volatility_Deviation_Percent'] < consts.get('volatility_max') else -1, axis=1)
    df['range_filter_condition'] = df.apply(lambda row: 1 if row['RF_BUY'] == True else -1 if row['RF_SELL'] == True else 0, axis=1)

    for template in conf_templates:
        template_signal = pd.Series(0, index=df.index)
        # print('tmp', template_signal.head(10))
        for ind_num, ind in enumerate(template['indicators']):
            full_condition_buy = (df[f"{ind}_condition"] == 1)
            full_condition_sell = (df[f"{ind}_condition"] == -1)
            if ind_num > 0:
                for i_shift in range(1, consts.get('prev_bars_count_allowance')):
                    full_condition_buy |= (df[f"{ind}_condition"].shift(i_shift) == 1)
                    full_condition_sell |= (df[f"{ind}_condition"].shift(i_shift) == -1)
            template_signal += full_condition_buy.astype(int) - full_condition_sell.astype(int)
        df[f"{template['name']}_signal"] = template_signal
        # Применение нормализации сигнала
        df[f"{template['name']}_signal"] = template_signal.apply(lambda x: 1 if x > 0 and x == template_signal.max() else -1 if x < 0 and x == template_signal.min() else 0)
        # print(df.head(100))

    return df


def plot_signals(df: pd.DataFrame, symbol: str, fig: Figure, row: int, col: int):
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
    if consts.get('rf_display_lines'):
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Bollinger_T_Low'], mode='lines', name=f'BB Trend Low', line=dict(color="rgba(255, 128, 128, 0.3)"), showlegend=sl), row=row, col=col)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Bollinger_T_High'], mode='lines', name=f'BB Trnd High', line=dict(color="rgba(100, 100, 255, 0.3)"), showlegend=sl), row=row, col=col)

    # Hband and lband
    if consts.get('bb_t_display_lines'):
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


app = Dash(__name__)


app.layout = html.Div([
    html.Div("Trading Signals for Cryptocurrencies", style={'textAlign': 'center', 'fontSize': 24, 'padding': '20px 0', 'color': 'white', 'fontWeight': 140}),
    dcc.Graph(id='main-graph'),
    dcc.Store(id='ohlcv-data'),
    dcc.Interval(id='interval-component', interval=settings['update_rate_minutes'] * 60*1000, n_intervals=0)
], style={'backgroundColor': '#171727', 'margin': 0, 'padding': 0})

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Dash</title>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
            }
            #root {
                height: 100%;
                width: 100%;
            }
        </style>
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body>
        
        <div id="root">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


@app.callback(
    Output('ohlcv-data', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_data(n_intervals):
    symbols = settings['symbols']
    data = generate_formatted_olhv(symbols)
    df = process_data(data)
    return df.to_dict('records')


@app.callback(
    Output('main-graph', 'figure'),
    Input('ohlcv-data', 'data'),
)
def update_graph(data):

    df = pd.DataFrame(data)
    symbols = settings['symbols']

    # Обновляем создание subplots для 2 колонок
    cols = 2
    rows = (len(symbols) * 2 + 2) // 2  # Расчет количества строк для двух столбцов, каждая монета занимает 2 строки (Price и MACD)
    subplot_titles = []
    for i in range(0, len(symbols), 2):
        if i + 1 < len(symbols):
            subplot_titles.extend([f"{symbols[i]} Price (x{i*2+1}, y{i*2+1})", f"{symbols[i + 1]} Price (x{i*2+2}, y{i*2+2})"])
            subplot_titles.extend([f"{symbols[i]} Help (x{i*2+3}, y{i*2+3})", f"{symbols[i + 1]} Help (x{i*2+4}, y{i*2+4})"])
        else:
            subplot_titles.extend([f"{symbols[i]} Price (x{i*2+1}, y{i*2+1})", "-"])
            subplot_titles.extend([f"{symbols[i]} Help (x{i*2+3}, y{i*2+3})", "-"])
    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=False, vertical_spacing=0.1 / rows, horizontal_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=[0.4 if i % 2 else 1 for i in range(rows)], column_widths=[0.5, 0.5]
    )

    fig.update_layout(
        paper_bgcolor="#171727",
        plot_bgcolor='#171727',
    )
    # print(df['timestamp'])

    df['int_timestamp'] = df['timestamp'].apply(lambda x: datetime.fromisoformat(x).timestamp())

    row = 1
    col = 1
    for index, symbol in enumerate(symbols):
        loguru.logger.debug(f"Processing symbol {symbol}")
        consts.set_symbol(symbol)
        df_symbol = df[df['symbol'] == symbol].copy().reset_index(drop=True)

        conf_templates = indicator_configs_template[:]
        for con in conf_templates:
            con['indicators'] = consts.get(con['name']).strip().split('+')

        df_symbol = calculate_indicators(df_symbol, conf_templates)

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

        for template in conf_templates:
            s_name = template['name']

            for i, l_row in last_n_records.iterrows():
                if l_row[f'{s_name}_signal'] == 1 and (symbol, s_name, l_row['int_timestamp']) not in unique_signals:
                    add_signal("buy", l_row['int_timestamp'], symbol, s_name, l_row['price'], df_symbol.iloc[-1]['price'])
                elif l_row[f'{s_name}_signal'] == -1 and (symbol, s_name, l_row['int_timestamp']) not in unique_signals:
                    add_signal("sell", l_row['int_timestamp'], symbol, s_name, l_row['price'], df_symbol.iloc[-1]['price'])
            # while pygame.mixer.get_busy():
            #     pygame.time.delay(10)

    fig.update_layout(
        height=settings['height'] * rows,
        dragmode='pan',  # Панорамирование по умолчанию
        width=settings['width'] * cols,
        plot_bgcolor='#171727',  # Цвет заднего фона графика
        margin=dict(t=40, b=40, l=10, r=10),  # Установка отступов
        hovermode="x unified",  # Режим наведения
        uirevision='constant',  # Оставаться в той же позиции при обновлении
    )

    # Отключение range slider для всех осей
    for axis_num in range(1, rows * cols + 1):
        fig.update_layout(**{
            f'xaxis{axis_num}_rangeslider_visible': False,
            f'xaxis{axis_num}_gridcolor': "#2b2f3b",
            f'yaxis{axis_num}_gridcolor': "#2b2f3b",
            f'yaxis{axis_num}_showspikes': True,
            f'xaxis{axis_num}_showspikes': True,
            f'yaxis{axis_num}_spikemode': 'across',
            f'xaxis{axis_num}_spikemode': 'across',
            f'yaxis{axis_num}_showline': True,
            f'xaxis{axis_num}_showline': True,
            # f'yaxis{axis_num}_spikesnap': 'cursor',
            # f'xaxis{axis_num}_spikesnap': 'cursor',
        })

    # Настройка осей для синхронного панорамирования парных графиков
    for i in range(1, len(symbols)*2, 4):
        if f'xaxis{i}' in fig['layout']:
            fig['layout'][f'xaxis{i+2}'].update(matches=f'x{i}')
        if f'xaxis{i+3}' in fig['layout']:
            fig['layout'][f'xaxis{i+3}'].update(matches=f'x{i+1}')

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
