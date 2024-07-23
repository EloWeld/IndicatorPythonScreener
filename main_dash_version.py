from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output

import loguru
import pandas as pd
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from core.price_data_aggregator import generate_formatted_olhv, process_data
from core.signal_service import add_signal
from core.symbol_processing import process_symbol
from core.utils import *

from pinepython.range_filter import RangeFilter
from core.config import *
import concurrent.futures


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


@app.callback(Output('main-graph', 'figure'), Input('ohlcv-data', 'data'),)
def update_graph(data):
    global is_browser_opened

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
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = []
    #     row = 1
    #     col = 1
    #     for index, symbol in enumerate(symbols):
    #         if index % 2 == 0 and index != 0:
    #             row += 2
    #         if index % 2 == 0:
    #             col = 1
    #         else:
    #             col = 2
    #         futures.append(executor.submit(process_symbol, df, symbol, fig, row, col))

    #     for future in concurrent.futures.as_completed(futures):
    #         fig = future.result()

    for index, symbol in enumerate(symbols):
        if index % 2 == 0 and index != 0:
            row += 2
        if index % 2 == 0:
            col = 1
        else:
            col = 2
        process_symbol(df, symbol, fig, row, col)

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
    open_url(f"http://127.0.0.1:{settings['service_port']}")
    app.run_server(debug=False, port=settings['service_port'])
