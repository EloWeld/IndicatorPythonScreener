## Пояснения к переменным

- `symbols` ——— Хранит информацию о символах которые будут отображены, лучше работает с чётным количеством
- `timeframe` ——— какой выкачиваем таймфрейм
- `last_n_bars` ——— сколько последних баров выкачиваем
- `atr_period` ——— период ATR
- `atr_mult` ——— множитель ATR
- `last_sync_time` ——— менять на год назад если хотим выкачать данные заново
- `last_interval_minutes` ——— через сколько минут обновляем данные
- `update_rate_minutes`  ——— через сколько минут обновляем интерфейс
- `height`  ——— Высота графиков одной монеты в пикселях
- `width` ——— Ширина графиков одной монеты в пикселях
- `is_using_spot` ——— Используем спот или фьючи
- `rf_ema_window` ——— Окно баров для сглаживатедя в RF
- `rf_ema_mult` ——— Множитель для RF
- `bb_t_window` ——— Окно BoilerBand отвечающего за **пробив**
- `bb_v_window` ——— Окно BoilerBand отвечающего за **волатильность**
- `bb_t_deviation` ——— Отклонение BoilerBand отвечающего за **пробив**
- `bb_v_deviation` ——— Отклонение BoilerBand отвечающего за **волатильность**

## Настройки поддерживающие кастомизацию по символам:
- `rf_ema_window`
- `rf_ema_mult`
- `bb_v_window`
- `bb_v_deviation`
- `bb_t_window`
- `bb_t_deviation`
- `atr_period`
- `atr_mult`
- `bb_dev_ma_window`
- `volatility_min`
- `volatility_max`