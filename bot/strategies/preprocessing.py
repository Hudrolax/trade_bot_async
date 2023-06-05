import pandas as pd
import copy
from decimal import Decimal, ROUND_HALF_UP
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

def count_zeros_after_decimal(value: float | int | str | Decimal) -> int:
    if isinstance(value, float):
        value = Decimal.from_float(value)
    elif isinstance(value, int):
        value = Decimal(value)
    elif isinstance(value, str):
        value = Decimal(value)
    elif not isinstance(value, Decimal):
        raise ValueError("Неподдерживаемый тип значения")

    decimal_tuple = value.as_tuple()
    exponent = decimal_tuple.exponent
    zeros_count = 0

    if exponent < 0:
        zeros_count = abs(exponent) - 1

    return int(zeros_count)


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
