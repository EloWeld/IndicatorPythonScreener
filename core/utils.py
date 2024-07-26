# Функция для генерации цвета на основе хэша
import hashlib
import os
import platform
import random


def generate_color(symbol):
    hash_object = hashlib.sha256(symbol.encode())
    hash_hex = hash_object.hexdigest()
    random.seed(int(hash_hex, 16))

    # Генерация HSL цвета с низкой насыщенностью
    h = random.randint(0, 360)
    s = 40
    l = 60

    return f'hsl({h}, {s}%, {l}%)'


def add_vertical_offset(y, offset):
    y_with_offset = y + (y * offset)
    return y_with_offset


def open_url(url):
    if platform.system() == 'Darwin':       # macOS
        os.system(f'open {url}')
    elif platform.system() == 'Windows':    # Windows
        os.system(f'start {url}')
    else:                                   # Linux variants
        os.system(f'xdg-open {url}')


def open_html_file(file_path):
    if platform.system() == 'Darwin':       # macOS
        os.system(f'open {file_path}')
    elif platform.system() == 'Windows':    # Windows
        os.system(f'start {file_path}')
    else:                                   # Linux variants
        os.system(f'xdg-open {file_path}')


def timeframe_to_seconds(timeframe):
    mapping = {
        '1m': 60,
        '2m': 60*2,
        '3m': 60*3,
        '4m': 60*4,
        '5m': 300,
        '10m': 600,
        '20m': 1200,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '1d': 86400
    }
    return mapping.get(timeframe, 60)
