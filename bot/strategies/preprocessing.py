import pandas as pd
import copy
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
