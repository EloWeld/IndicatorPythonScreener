
import json
import os
import sys

import loguru


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


def save_cached(setts, prices):
    with open(resource_path('cached_prices.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(prices))
    with open(resource_path('ind_config.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(setts, indent=4, default=str))


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
