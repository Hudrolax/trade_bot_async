import pandas as pd
import numpy as np


def add_prefix(column_name, prefix) -> str:
    return f"{prefix}_{column_name}" if prefix else column_name

def bollinger_bands(df, price='close', period=20, deviation=2.0, prefix='') -> tuple[pd.DataFrame, list]:
    """
    Calculate the Bollinger Bands for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the historical data with price columns.
        price (str, optional): Name of the price column to use for calculation. Defaults to 'close'.
        period (int, optional): Period for the Bollinger Bands calculation. Defaults to 20.
        deviation (int, optional): Number of standard deviations for the upper and lower bands. Defaults to 2.
        prefix (str, optional): Prefix to add to the column names. Defaults to an empty string.

    Returns (tuple):
        pd.DataFrame: Dataframe with the added Bollinger Bands columns containing the calculated values.
        [list of col names]
    """
    # Copy the original dataframe
    df_copy = df.copy()
    df_copy[price] = df_copy[price].astype(np.float64)

    # Add the prefix to the price column
    price_col = add_prefix(price, prefix)

    # Calculate the moving average and standard deviation
    df_copy[add_prefix('ma', prefix)] = df_copy[price_col].rolling(window=period).mean()
    df_copy[add_prefix('std', prefix)] = df_copy[price_col].rolling(window=period).std()

    # Calculate the Bollinger Bands
    upper_name = add_prefix('bb_upper', prefix)
    middle_name = add_prefix('bb_middle', prefix)
    lower_name = add_prefix('bb_lower', prefix) 
    df_copy[upper_name] = df_copy[add_prefix('ma', prefix)] + (df_copy[add_prefix('std', prefix)] * deviation)
    df_copy[middle_name] = df_copy[add_prefix('ma', prefix)]
    df_copy[lower_name] = df_copy[add_prefix('ma', prefix)] - (df_copy[add_prefix('std', prefix)] * deviation)

    # Remove the temporary columns
    df_copy.drop(columns=[
        add_prefix('ma', prefix),
        add_prefix('std', prefix),
    ], inplace=True)

    return df_copy, [upper_name, middle_name, lower_name]

def rsi(df, price='close', period=14, prefix='') -> tuple[pd.DataFrame, list]:
    """
    Calculate the Relative Strength Index (RSI) indicator for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the historical data with price columns.
        price (str, optional): Name of the price column to use for calculation. Defaults to 'close'.
        period (int, optional): Period for the RSI calculation. Defaults to 14.
        prefix (str, optional): Prefix to add to the column names. Defaults to an empty string.

    Returns (tuple):
        pd.DataFrame: Dataframe with the added RSI column containing the calculated RSI values.
        [list of col names]
    """
    # Copy the original dataframe
    df_copy = df.copy()
    df_copy[price] = df_copy[price].astype(np.float64)

    # Add the prefix to the price column
    price_col = add_prefix(price, prefix)

    # Calculate the price change
    df_copy[add_prefix('price_change', prefix)] = df_copy[price_col].diff()

    # Calculate the gains and losses
    df_copy[add_prefix('gain', prefix)] = df_copy[add_prefix('price_change', prefix)].where(df_copy[add_prefix('price_change', prefix)] > 0, 0)
    df_copy[add_prefix('loss', prefix)] = -df_copy[add_prefix('price_change', prefix)].where(df_copy[add_prefix('price_change', prefix)] < 0, 0)

    # Calculate the average gains and losses
    df_copy[add_prefix('avg_gain', prefix)] = df_copy[add_prefix('gain', prefix)].rolling(window=period).mean()
    df_copy[add_prefix('avg_loss', prefix)] = df_copy[add_prefix('loss', prefix)].rolling(window=period).mean()

    # Calculate the Relative Strength (RS)
    df_copy[add_prefix('rs', prefix)] = df_copy[add_prefix('avg_gain', prefix)] / df_copy[add_prefix('avg_loss', prefix)]

    # Calculate the Relative Strength Index (RSI)
    rsi_name = add_prefix('rsi', prefix)
    df_copy[rsi_name] = 100 - (100 / (1 + df_copy[add_prefix('rs', prefix)]))

    # Remove the temporary columns
    df_copy.drop(columns=[
        add_prefix('price_change', prefix),
        add_prefix('gain', prefix),
        add_prefix('loss', prefix),
        add_prefix('avg_gain', prefix),
        add_prefix('avg_loss', prefix),
        add_prefix('rs', prefix)
    ], inplace=True)

    return df_copy, [rsi_name]

def moving_average(df, price='close', period=14, method='simple', prefix='') -> tuple[pd.DataFrame, list]:
    """
    Calculate the moving average for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the historical data with price columns.
        price (str, optional): Name of the price column to use for calculation. Defaults to 'close'.
        period (int, optional): Period for the moving average calculation. Defaults to 14.
        method (str, optional): Method for calculating the moving average ('simple' or 'exponential'). Defaults to 'simple'.
        prefix (str, optional): Prefix to add to the column names. Defaults to an empty string.

    Returns (tuple):
        pd.DataFrame: Dataframe with the added moving average column containing the calculated moving average values.
        [list of col names]
    """
    # Copy the original dataframe
    df_copy = df.copy()
    df_copy[price] = df_copy[price].astype(np.float64)

    # Add the prefix to the price column
    price_col = add_prefix(price, prefix)

    # Calculate the moving average based on the specified method
    col_name = add_prefix(f'ma{period}', prefix)
    if method.lower() == 'simple':
        df_copy[col_name] = df_copy[price_col].rolling(window=period).mean()
    elif method.lower() == 'exponential':
        df_copy[col_name] = df_copy[price_col].ewm(span=period, adjust=False).mean()
    else:
        raise ValueError("Invalid method. Available methods: 'simple', 'exponential'")

    return df_copy, [col_name]

def average_true_range(df, high='high', low='low', close='close', period=14, prefix='') -> tuple[pd.DataFrame, list]:
    """
    Calculate the Average True Range (ATR) indicator for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the historical data with high, low, and close columns.
        high (str, optional): Name of the high price column. Defaults to 'high'.
        low (str, optional): Name of the low price column. Defaults to 'low'.
        close (str, optional): Name of the close price column. Defaults to 'close'.
        period (int, optional): Period for the ATR calculation. Defaults to 14.
        prefix (str, optional): Prefix to add to the column names. Defaults to an empty string.

    Returns (tupe):
        pd.DataFrame: Dataframe with the added ATR column containing the calculated ATR values.
        [list of col names]
    """
    # Copy the original dataframe
    df_copy = df.copy()
    df_copy[high] = df_copy[high].astype(np.float64)
    df_copy[low] = df_copy[low].astype(np.float64)
    df_copy[close] = df_copy[close].astype(np.float64)

    # Add the prefix to the high, low, and close columns
    high_col = add_prefix(high, prefix)
    low_col = add_prefix(low, prefix)
    close_col = add_prefix(close, prefix)

    # Calculate the true range
    tr_col = add_prefix('tr', prefix)
    df_copy[tr_col] = df_copy[high_col] - df_copy[low_col]
    df_copy[tr_col] = df_copy[[tr_col, (df_copy[high_col] - df_copy[close_col]).abs(), (df_copy[low_col] - df_copy[close_col]).abs()]].max(axis=1)

    # Calculate the Average True Range (ATR)
    atr_col = add_prefix('atr', prefix)
    df_copy[atr_col] = df_copy[add_prefix('tr', prefix)].rolling(window=period).mean()

    # Remove the temporary column
    df_copy.drop(columns=[add_prefix('tr', prefix)], inplace=True)

    return df_copy, [atr_col]


def macd(df, price='close', short_period=12, long_period=26, signal_period=9, prefix='') -> tuple[pd.DataFrame, list]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the historical data with price columns.
        price (str, optional): Name of the price column to use for calculation. Defaults to 'close'.
        short_period (int, optional): Period for the short-term EMA. Defaults to 12.
        long_period (int, optional): Period for the long-term EMA. Defaults to 26.
        signal_period (int, optional): Period for the MACD signal line. Defaults to 9.
        prefix (str, optional): Prefix to add to the column names. Defaults to an empty string.

    Returns (tuple):
        pd.DataFrame: Dataframe with the added MACD columns containing the calculated MACD values.
        [list of col names]
    """
    # Copy the original dataframe
    df_copy = df.copy()
    df_copy[price] = df_copy[price].astype(np.float64)

    # Add the prefix to the price column
    price_col = add_prefix(price, prefix)

    # Calculate the short-term and long-term Exponential Moving Averages (EMAs)
    short_ema_name = add_prefix('short_ema', prefix)
    long_ema_name = add_prefix('long_ema', prefix)
    df_copy[short_ema_name] = df_copy[price_col].ewm(span=short_period, adjust=False).mean()
    df_copy[long_ema_name] = df_copy[price_col].ewm(span=long_period, adjust=False).mean()

    # Calculate the MACD line
    macd_name = add_prefix('macd', prefix)
    df_copy[macd_name] = df_copy[short_ema_name] - df_copy[long_ema_name]

    # Calculate the MACD signal line
    macd_signal_name = add_prefix('macd_signal', prefix)
    df_copy[macd_signal_name] = df_copy[macd_name].ewm(span=signal_period, adjust=False).mean()

    # Calculate the MACD histogram
    macd_hist_name = add_prefix('macd_histogram', prefix)
    df_copy[macd_hist_name] = df_copy[macd_name] - df_copy[macd_signal_name]

    # Remove the temporary columns
    df_copy.drop(columns=[short_ema_name, long_ema_name], inplace=True)

    return df_copy, [macd_name, macd_signal_name, macd_hist_name]

def obv(df, close='close', volume='vol', prefix='') -> tuple[pd.DataFrame, list]:
    """
    Calculate the On Balance Volume (OBV) indicator for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the historical data with close and volume columns.
        close (str, optional): Name of the close price column to use for calculation. Defaults to 'close'.
        volume (str, optional): Name of the volume column to use for calculation. Defaults to 'volume'.
        prefix (str, optional): Prefix to add to the column names. Defaults to an empty string.

    Returns:
        pd.DataFrame: Dataframe with the added OBV column containing the calculated OBV values.
    """
    # Copy the original dataframe
    df_copy = df.copy()
    df_copy[close] = df_copy[close].astype(np.float64)
    df_copy[volume] = df_copy[volume].astype(np.float64)

    # Add the prefix to the close and volume columns
    close_col = add_prefix(close, prefix)
    volume_col = add_prefix(volume, prefix)

    # Calculate the daily returns
    df_copy[add_prefix('return', prefix)] = df_copy[close_col].diff()

    # Calculate the OBV
    df_copy[add_prefix('obv', prefix)] = np.where(df_copy[add_prefix('return', prefix)] > 0, df_copy[volume_col], 
                                                  np.where(df_copy[add_prefix('return', prefix)] < 0, -df_copy[volume_col], 0)).cumsum()

    # Remove the temporary columns
    df_copy.drop(columns=[
        add_prefix('return', prefix),
    ], inplace=True)

    return df_copy, [add_prefix('obv', prefix)]
