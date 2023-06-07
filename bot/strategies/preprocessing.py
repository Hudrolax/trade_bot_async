import pandas as pd
import copy
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation, getcontext
from .indicators import (
    bollinger_bands,
    rsi,
    moving_average,
    average_true_range,
    macd,
    obv,
)


indicator_func = {
    'bb': bollinger_bands,
    'rsi': rsi,
    'ma': moving_average,
    'atr': average_true_range,
    'macd': macd,
    'obv': obv,
}

getcontext().prec = 10 # Set precision of Decimal to maximum for float64

def count_significant_digits(n):
    if isinstance(n, int):
        return 0
    elif isinstance(n, float):
        # We limit the precision to 10 digits after the dot.
        n_str = f'{n:.10f}'.rstrip('0').rstrip('.')
    elif isinstance(n, str):
        try:
            # Attempt to convert string to Decimal for better precision handling
            n = Decimal(n)
            n_str = str(n.normalize())
        except InvalidOperation:
            # If the conversion fails (for example, due to a string like '0.1 + 0.2'),
            # fall back to using the string as is
            n_str = n
    elif isinstance(n, Decimal):
        n_str = str(n.normalize())
    else:
        raise TypeError("Unsupported type")

    if '.' not in n_str:
        return 0
    else:
        return len(n_str) - n_str.index('.') - 1


def float_to_decimal(value: Decimal | float, decimal_places: int) -> Decimal:
    decimal_places = int(decimal_places)
    decimal_value = Decimal(value)
    rounding = Decimal(10) ** (-decimal_places)
    return decimal_value.quantize(rounding, ROUND_HALF_UP)


def calculate_indicators(klines: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    """Calculate indicators for klines.

    Args:
        klines (pd.DataFrame): klines
        kwargs (dict):
            **{kwargs}: specify kwargs for indicator (like price, period, deviation, etc...)

    Returns:
        pd.DataFrame - dataframe with indicators
    """
    kwargs = copy.deepcopy(kwargs)
    klines = klines.copy()
    for key in kwargs.keys():
        value = kwargs[key]

        klines, cols = indicator_func[key](klines, **value)

    return klines.dropna().reset_index(drop=True)
