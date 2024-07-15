# Функция для генерации цвета на основе хэша
import hashlib
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
