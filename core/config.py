
import json
import os
import sys

import loguru
from concurrent.futures import ThreadPoolExecutor


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


settings = {}
cached_prices = {}
unique_signals = set()


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


# LOAD SETTINGS
with open(resource_path('ind_config.json'), 'r', encoding='utf-8') as f:
    settings = json.loads(f.read())
# LOAD SETTINGS
try:
    with open(resource_path('last_signals.json'), 'r', encoding='utf-8') as f:
        unique_signals_list = json.loads(f.read())

        for x in unique_signals_list:
            unique_signals.add(tuple(x))
except Exception as e:
    loguru.logger.error(str(e))

with open(resource_path('cached_prices.json'), 'r', encoding='utf-8') as f:
    cached_prices = json.loads(f.read())


def save_signals():
    with open(resource_path('last_signals.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps([list(x) for x in list(unique_signals)], indent=4))

# SAVE PRICES TO FILE (CACHE)


def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4, default=str))


def save_cached(setts, prices):
    with ThreadPoolExecutor() as executor:
        executor.submit(save_json, resource_path('cached_prices.json'), prices)
        executor.submit(save_json, resource_path('ind_config.json'), setts)


class Consts:
    def __init__(self) -> None:
        self._symbol = None

    def set_symbol(self, symbol: str):
        self._symbol = symbol

    def get(self, const_name: str):
        try:
            if self._symbol is None:
                return settings[const_name]
            elif self._symbol in settings['spec_settings'] and const_name in settings['spec_settings'][self._symbol]:
                return settings['spec_settings'][self._symbol][const_name]
            else:
                return settings[const_name]
        except Exception as e:
            loguru.logger.error(f"Const name \"{const_name}\" not found! Current symbol: {self._symbol}; Error: {e};")


consts = Consts()
