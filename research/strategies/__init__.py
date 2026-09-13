from .buy_hold import signal as buy_hold
from .ema import signal as ema
from .rsi import signal as rsi
from .macd import signal as macd
from .bollinger import signal as bollinger
from .momentum import signal as momentum

STRATEGIES = {
    "BuyHold": buy_hold, "EMA": ema, "RSI": rsi,
    "MACD": macd, "Bollinger": bollinger, "Momentum": momentum
}
