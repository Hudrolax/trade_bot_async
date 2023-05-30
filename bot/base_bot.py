import asyncio
import signal
import pandas as pd
import logging
from bot.binance_um_futures import (
    wss_klines,
    get_klines,
    run_user_data_stream,
    get_account_info,
    get_market_info,
)
from bot.data_class import Strategy
from decimal import Decimal

logger = logging.getLogger(__name__)

# The collumns for klines dataframe
klines_cols = ['open_time', 'open', 'high', 'low', 'close',
               'vol', 'close_time', 'qa_vol', 'trades',]
addition_klines_cols = ['symbol', 'tf', 'market']

# wb - wallet balance, cw - cross wallet balance, bc = balance except pnl and cpomission
balances_cols = ['market', 'asset', 'wb', 'cw', 'ab']
positions_cols = ['symbol', 'amount', 'entry_price', 'pnl', 'side']
market_info_cols = ['market', 'symbol', 'status', 'baseAsset',
                    'quoteAsset', 'pricePrecision', 'quantityPrecision', 'baseAssetPrecision', 'quotePrecision',
                    'tickSize', 'minQty', 'maxQty', 'stepSize', 'minNotional']


def update_or_insert(df1: pd.DataFrame, new_row: dict, keys: list) -> pd.DataFrame:
    """The function makes insert or update a dataframe

    Args:
        df1 (pd.DataFrame): the dataframe
        new_row (dict): new row in dict format
        keys (list): list of primary keys

    Returns:
        pd.DataFrame: updated dataframe
    """
    df1.set_index(keys, inplace=True)
    df2 = pd.DataFrame([new_row])
    df2.set_index(keys, inplace=True)

    # Create a dictionary of dtypes for each column in df1
    dtype_dict = df1.dtypes.to_dict()

    # Cast columns in df2 to match dtypes in df1
    for col, dtype in dtype_dict.items():
        if col in df2.columns:
            df2[col] = df2[col].astype(dtype)

    # Update df1 with df2 values
    df1.update(df2)

    # Concatenate dataframes and remove duplicates (keeps the first occurrence)
    result = pd.concat([df1, df2]).loc[~pd.concat(
        [df1, df2]).index.duplicated(keep='first')]

    # Reset the index
    result.reset_index(inplace=True)

    return result


def change_type_df(df: pd.DataFrame) -> pd.DataFrame:
    """The function normalizes the value types in the dataframe."""
    # change float types
    for col in ['open', 'high', 'low', 'close', 'vol', 'qa_vol']:
        if col in df.columns:
            df[col] = df[col].astype('float64')

    # change int types
    for col in ['trades']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # change date types
    if 'open_time' in df.columns and df['open_time'].dtype == int:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

    if 'close_time' in df.columns and df['close_time'].dtype == int:
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    return df


class BaseBot:
    def __init__(self, strategies: list[dict]) -> None:
        self.stop = asyncio.Event()
        self.klines_lock = asyncio.Lock()
        self.balances_lock = asyncio.Lock()
        self.positions_lock = asyncio.Lock()
        self.market_lock = asyncio.Lock()
        self.strategies = []
        for strategy in strategies:
            self.strategies.append(Strategy(**strategy))

        self.klines: pd.DataFrame = pd.DataFrame(
            [], columns=[*klines_cols, *addition_klines_cols])

        self.balances: pd.DataFrame = pd.DataFrame([], columns=balances_cols)
        self.positions: pd.DataFrame = pd.DataFrame([], columns=positions_cols)
        self.market_info: pd.DataFrame = pd.DataFrame(
            [], columns=market_info_cols)
        logger.info('****** Trade BOT ******')

    @staticmethod
    def shrink_klines(df: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
        """The function shrink the dataframe"""
        mask = (df['symbol'] == strategy.symbol) & (
            df['tf'] == strategy.tf) & (df['market'] == strategy.market)
        min_time = df[mask].sort_values(
            by='open_time').iloc[-strategy.window]['open_time']
        df = df.drop(df[mask & (df['open_time'] < min_time)].index)
        df = df.sort_values(
            by=['market', 'symbol', 'tf', 'open_time']).reset_index(drop=True)
        return df

    def stop_handler(self, signum, frame):
        """Handler for stop signal"""
        logger.info('Received stop signal')
        self.stop.set()

    async def last_price(self, market: str, symbol: str) -> float:
        """The function returns a last price of market/symbol

        Args:
            market (str): market name (see in data_class.py)
            symbol (str): symbol name

        Returns:
            float: last price
        """
        async with self.klines_lock:
            mask = (self.klines['market'] == market) & (
                self.klines['symbol'] == symbol)
            last_price = self.klines[mask].sort_values(
                by='open_price').iloc[-1]['close']
        return last_price

    async def user_data_handler(self, data: dict):
        if data['e'] == "ACCOUNT_UPDATE":
            # update balances
            async with self.balances_lock:
                for asset in data['a']['B']:
                    mask = (
                        self.balances['market'] == 'um-futures-cross') & (self.balances['asset'] == asset['a'])
                    wb = float(asset['wb'])
                    cw = float(asset['cw'])

                    if mask.any():
                        self.balances.loc[mask, 'wb'] = wb
                        self.balances.loc[mask, 'cw'] = wb
                    else:
                        row = [['um-futures-cross', asset['a'], wb, cw]]
                        new_row = pd.DataFrame(row, columns=balances_cols)
                        self.balances = pd.concat(
                            [self.balances, new_row], ignore_index=True)
                self.balances.to_csv('balances.csv', index=False)
            # update positions
            async with self.positions_lock:
                for position in data['a']['P']:
                    mask = self.positions['symbol'] == position['s']
                    amount = Decimal(position['pa'])
                    entry_price = Decimal(position['ep'])
                    pnl = float(position['up'])
                    async with self.klines_lock:
                        klines_mask = (
                            self.klines['market'] == 'um-futures-cross') & (self.klines['symbol'] == position['s'])
                        prices = self.klines[klines_mask].sort_values(
                            by='open_time')
                        if len(prices) == 0:
                            continue  # have no this symbol / market on strategy
                        last_price = prices.iloc[-1]['close']
                    side = 'BUY'
                    if last_price < entry_price and float(position['cr']) > 0:
                        side = 'SELL'

                    primary_keys = ['symbol']
                    row = dict(
                        symbol=position['s'],
                        amount=amount,
                        entry_price=entry_price,
                        pnl=pnl,
                        side=side,
                    )
                    self.positions = update_or_insert(
                        self.positions, row, primary_keys)

                self.positions.to_csv('positions.csv', index=False)

    async def tick_handler(self, strategy: Strategy, data: dict):
        """ Handler receive a tick from Binance

        Args:
            strategy (Strategy): Strategy object
            data (dict): raw data from Binance
        """
        symbol = strategy.symbol
        tf = strategy.tf
        market = strategy.market
        kline = data['k']
        # preprocess klines info
        async with self.klines_lock:
            row = dict(
                symbol=symbol,
                tf=tf,
                market=market,
                open_time=pd.to_datetime(kline['t'], unit='ms'),
                close_time=pd.to_datetime(kline['T'], unit='ms'),
                open=float(kline['o']),
                high=float(kline['h']),
                low=float(kline['l']),
                close=float(kline['c']),
                vol=float(kline['v']),
                qa_vol=float(kline['q']),
                trades=float(kline['n']),
            )
            primary_keys = ['symbol', 'tf', 'market', 'open_time']
            self.klines = update_or_insert(self.klines, row, keys=primary_keys)

            self.klines = self.shrink_klines(self.klines, strategy)
            self.klines.to_csv('klines.csv', index=False)

            mask = (self.klines['symbol'] == symbol) & (
                self.klines['tf'] == tf) & (self.klines['market'] == market)
            df = self.klines[mask].sort_values(
                by='open_time', ignore_index=True).copy()

        # get account info
        # update balances

        # apply a strategy
        await self.on_tick(strategy=strategy, klines=df)

    async def update_accaunt_info(self, market: str) -> None:
        """The function update account info for a specific market.

        Args:
            market (str): market name
        """
        info: dict = await get_account_info()
        # update balances
        async with self.balances_lock:
            for asset in info['assets']:
                primary_keys = ['market', 'asset']
                row = dict(
                    market=market,
                    asset=asset['asset'],
                    wb=float(asset['walletBalance']),
                    cw=float(asset['crossWalletBalance']),
                    ab=float(asset['availableBalance']),
                )
                self.balances = update_or_insert(
                    self.balances, row, primary_keys)

        # update positions
        async with self.positions_lock:
            for position in info['positions']:
                primary_keys = ['symbol']
                side = 'BUY'
                if float(position['entryPrice']) > self.last_price(market, position['symbol']) \
                        and float(position['unrealizedProfit']) > 0:
                    side = 'SELL'
                row = dict(
                    symbol=position['symbol'],
                    amount=Decimal(position['positionAmt']),
                    entry_price=Decimal(position['entryPrice']),
                    pnl=float(position['unrealizedProfit']),
                    side=side,
                )
                self.positions = update_or_insert(
                    self.positions, row, primary_keys)

    async def on_tick(self, strategy: Strategy, klines: pd.DataFrame):
        """ Not implemented stategy handler

        Args:
            strategy (Strategy): a strategy object
            klines (pd.DataFrame): klines df of strategy symbol
        """
        raise NotImplementedError

    async def download_klines(self, strategy: Strategy):
        """Function download and preprocess raw klines from Binance"""
        klines_raw = await get_klines(strategy.symbol, strategy.tf, self.stop, limit=strategy.window)
        df = pd.DataFrame(klines_raw).drop([9, 10, 11], axis=1)
        df.columns = klines_cols
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['symbol'] = strategy.symbol
        df['tf'] = strategy.tf
        df['market'] = strategy.market
        df = change_type_df(df)
        async with self.klines_lock:
            self.klines = pd.concat([self.klines, df], ignore_index=True)
            self.klines = self.klines.drop_duplicates()

    async def get_market_info(self, market: str) -> None:
        if market == 'um-futures-cross' or market == 'um-futures':
            raw_data = await get_market_info(self.stop)
        else:
            raise ValueError(f'Wrong market name: {market}')

        for symbol in raw_data['symbols']:
            row = dict(
                market=market,
                symbol=symbol['symbol'],
                status=symbol['status'],
                baseAsset=symbol['baseAsset'],
                quoteAsset=symbol['quoteAsset'],
                pricePrecision=Decimal(symbol['pricePrecision']),
                quantityPrecision=Decimal(symbol['quantityPrecision']),
                baseAssetPrecision=Decimal(symbol['baseAssetPrecision']),
                quotePrecision=Decimal(symbol['quotePrecision']),
            )
            for filtr in symbol['filters']:
                if filtr['filterType'] == 'PRICE_FILTER':
                    row['tickSize'] = Decimal(filtr['tickSize'])
                elif filtr['filterType'] == 'LOT_SIZE':
                    row['maxQty'] = Decimal(filtr['maxQty'])
                    row['minQty'] = Decimal(filtr['minQty'])
                    row['stepSize'] = Decimal(filtr['stepSize'])
                elif filtr['filterType'] == 'MIN_NOTIONAL':
                    row['minNotional'] = Decimal(filtr['notional'])
            async with self.market_lock:
                self.market_info = update_or_insert(
                    self.market_info, row, ['market', 'symbol'])
                # self.market_info.to_csv('market_info.csv', index=False)

    async def update_market_info(self, market: str) -> None:
        while not self.stop.is_set():
            try:
                await self.get_market_info(market)
            except ValueError as ex:
                logger.critical(ex)
                raise ex
            except Exception as ex:
                logger.error(ex)
            finally:
                await asyncio.sleep(60 * 60)  # one hour

    async def run_strategy(self, strategy: Strategy):
        """Function runs an infinity loop for every strategy"""
        logger.info(
            f"Run strategy {strategy.name} for {strategy.symbol}_{strategy.tf} on {strategy.market}")
        try:
            # download klines history before run the strategy
            await self.download_klines(strategy)

            # run tick stream
            await wss_klines(self.tick_handler, strategy, self.stop)
        except Exception as e:
            logger.critical(e)
            raise e

    async def run(self):
        """The main runbot method"""
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self.stop_handler, loop, self)
        loop.add_signal_handler(signal.SIGTSTP, self.stop_handler, loop, self)
        loop.add_signal_handler(signal.SIGTERM, self.stop_handler, loop, self)

        tasks = []
        # run update market info
        tasks.append(asyncio.create_task(
            self.update_market_info('um-futures-cross')))

        # run user data stream
        tasks.append(asyncio.create_task(
            run_user_data_stream(self.user_data_handler, self.stop)))

        # run strategies
        for strategy in self.strategies:
            tasks.append(asyncio.create_task(self.run_strategy(strategy)))

        await self.stop.wait()
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks)
