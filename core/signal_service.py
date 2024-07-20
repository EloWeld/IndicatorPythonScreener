
import json
import traceback
import loguru
import pygame
import requests
from datetime import datetime

from core.config import *
from core.utils import *

# Инициализация микшера Pygame
pygame.mixer.init()

sound_up = pygame.mixer.Sound(resource_path('sound_up.mp3'))
sound_down = pygame.mixer.Sound(resource_path('sound_down.mp3'))


def add_signal(side: str, timestamp, symbol, name, signal_price, currenct_price):
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

    link = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}" if settings['is_using_spot'] else f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}.P"

    message = f"""{'🟢' if side == 'buy' else '🔴'} {symbol} : <code>{side}</code>

Old/New prices: {signal_price}/{currenct_price}
<b>{(currenct_price - signal_price)/signal_price * 100:.2f}%</b>

{link}

Signal Time: <code>{datetime.fromtimestamp(timestamp / 1000).strftime('%Y.%m.%d %H:%M:%S')}</code>
Symbol: <code>{symbol}</code>
Description: <code>{name}</code>"""
    telegram_api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    for user_id in user_ids:
        try:
            response = requests.post(
                telegram_api_url,
                data={'chat_id': user_id, 'text': message, 'parse_mode': 'html', 'disable_web_page_preview': True},

            )
            if response.status_code != 200:
                loguru.logger.error(f"Failed to send message to {user_id}. Status code: {response.status_code}, response: {response.text}")
        except Exception as e:
            loguru.logger.error(f"Error while sending Telegram message! Error: {e}; Traceback: {traceback.format_exc()}")

    # Send webhook
    webhook_data_str = "?"
    category = name
    try:
        webhook_data_str = json.dumps(consts.get(f'{category}_{side}_webhook_data'))
        webhook_data_str = webhook_data_str.replace('{{symbol}}', symbol).replace('{{side}}', side).replace('{{desc}}', name)
        webhook_data = json.loads(webhook_data_str)

        resp = requests.post(consts.get(f'{category}_{side}_webhook_url'), json=webhook_data)
        if resp.status_code != 200:
            loguru.logger.error(f"Not success status code while sending webhook! Status code: {resp.status_code}, data: {resp.text}")
    except Exception as e:
        loguru.logger.error(f"Error while sending webhook! sending_data: {webhook_data_str}; Error: {e}; Traceback: {traceback.format_exc()}")
