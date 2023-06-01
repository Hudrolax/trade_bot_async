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
from bot.errors import OpenOrderError, CloseOrderError, TickHandleError
from decimal import Decimal

logger = logging.getLogger(__name__)

# The collumns for klines dataframe
klines_cols = ['open_time', 'open', 'high', 'low', 'close',
               'vol', 'close_time', 'qa_vol', 'trades',]
addition_klines_cols = ['symbol', 'tf', 'market']

# wb - wallet balance, cw - cross wallet balance, bc = balance except pnl and cpomission
balances_cols = ['market', 'asset', 'wb', 'cw', 'ab']
positions_cols = ['market', 'symbol', 'amount', 'entry_price', 'pnl', 'side']
orders_cols = ['market', 'symbol', 'side', 'type', 'quantity',
               'price', 'average_price', 'status', 'id', 'client_id', 'trade_time']
market_info_cols = ['market', 'symbol', 'status', 'baseAsset',
                    'quoteAsset', 'pricePrecision', 'quantityPrecision', 'baseAssetPrecision', 'quotePrecision',
                    'tickSize', 'minQty', 'maxQty', 'stepSize', 'minNotional']


def update_or_insert(df1: pd.DataFrame, new_row: dict | pd.DataFrame, keys: list) -> pd.DataFrame:
    """The function makes insert or update a dataframe

    Args:
        df1 (pd.DataFrame): the dataframe
        new_row (dict): new row in dict format
        keys (list): list of primary keys

    Returns:
        pd.DataFrame: updated dataframe
    """
    df1.set_index(keys, inplace=True)
    if isinstance(new_row, dict):
        df2 = pd.DataFrame([new_row])
    elif isinstance(new_row, pd.DataFrame):
        df2 = new_row
    else:
        raise TypeError('new_row shoult be a dict or pd.Dataframe')
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
        self.orders_lock = asyncio.Lock()
        self.market_lock = asyncio.Lock()
        self.strategies = []
        for strategy in strategies:
            self.strategies.append(Strategy(**strategy))

        self.klines: pd.DataFrame = pd.DataFrame(
            [], columns=[*klines_cols, *addition_klines_cols])

        self.balances: pd.DataFrame = pd.DataFrame([], columns=balances_cols)
        self.positions: pd.DataFrame = pd.DataFrame([], columns=positions_cols)
        self.orders: pd.DataFrame = pd.DataFrame([], columns=orders_cols)
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

    async def get_strategy_positions(self, strategy: Strategy) -> pd.DataFrame:
        """The function gets positions for a certain strategy"""
        async with self.positions_lock:
            mask = (self.positions['market'] == strategy.market) & (
                self.positions['symbol'] == strategy.symbol)
            return self.positions[mask].copy()

    async def last_price(self, market: str, symbol: str) -> float:
        """The function returns a last price of market/symbol

        Args:
            market (str): market name (see in data_class.py)
            symbol (str): symbol name

        Returns:
            float: last price
        """
        logger = logging.getLogger('last_price')
        async with self.klines_lock:
            try:
                mask = (self.klines['market'] == market) & (
                    self.klines['symbol'] == symbol)
                last_price = self.klines[mask].sort_values(
                    by='open_time').iloc[-1]['close']
            except Exception as ex:
                logger.critical(ex)
                raise ex
        return last_price

    async def user_data_handler(self, data: dict):
        """The function opdates account and ositions data by user data stream

        Args:
            data (dict): raw data from Binance
        """
        if data['e'] == "ACCOUNT_UPDATE":
            # update balances
            async with self.balances_lock:
                for asset in data['a']['B']:
                    mask = (
                        self.balances['market'] == 'um-futures-cross') & (self.balances['asset'] == asset['a'])
                    wb = float(asset['wb'])
                    cw = float(asset['cw'])
                    primary_keys = ['market', 'asset']
                    old_ab = self.balances[mask].iloc[-1]['ab']
                    row = dict(
                        market='um-futures-cross',
                        asset=asset['a'],
                        wb=wb,
                        cw=cw,
                        ab=old_ab,
                    )
                    self.balances = update_or_insert(
                        self.balances, row, primary_keys)

                self.balances.to_csv('balances.csv', index=False)
            # update positions
            async with self.positions_lock:
                for position in data['a']['P']:
                    mask = self.positions['symbol'] == position['s']
                    amount = Decimal(position['pa'])
                    entry_price = Decimal(position['ep'])
                    pnl = float(position['up'])
                    side = 'BUY'
                    if await self.last_price(market='um-futures-cross', symbol=position['s']) < entry_price \
                            and float(position['cr']) > 0:
                        side = 'SELL'

                    primary_keys = ['symbol']
                    row = dict(
                        market='um-futures-cross',
                        symbol=position['s'],
                        amount=amount,
                        entry_price=entry_price,
                        pnl=pnl,
                        side=side,
                    )
                    self.positions = update_or_insert(
                        self.positions, row, primary_keys)

                self.positions.to_csv('positions.csv', index=False)

        elif data['e'] == 'ORDER_TRADE_UPDATE':
            orders_cols = ['market', 'symbol', 'side', 'type', 'quantity',
               'price', 'average_price', 'status', 'id', 'client_id', 'trade_time']
            order = data['o']
            row = dict(
                market='um-futures-cross',
                symbol=order['s'],
                side=order['S'],
                type=order['o'],
                quantity=Decimal(order['q']),
                price=Decimal(order['p']),
                average_price=Decimal(order['ap']),
                status=order['X'],
                id=order['i'],
                client_id=order['c'],
                trade_time=pd.to_datetime(order['T'], unit='ms'),
            )
            primary_keys = ['market', 'symbol', 'id']
            async with self.orders_lock:
                self.orders = update_or_insert(self.orders, row, primary_keys)
                self.orders.to_csv('orders.csv', index=False)

    async def tick_handler(self, strategy: Strategy, data: dict) -> None:
        """ Handler receive a tick from Binance

        Args:
            strategy (Strategy): Strategy object
            data (dict): raw data from Binance
        """
        symbol = strategy.symbol
        tf = strategy.tf
        market = strategy.market
        kline = data['k']
        is_kline_closed: bool = kline['x']
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

        # apply a strategy
        try:
            await self.on_tick(strategy=strategy, klines=df, is_kline_closed=is_kline_closed)
        except TickHandleError as ex:
            logger.critical(
                f'Unexpected tick error. NEED REFACTOR CODE!!! err: {ex}')
            raise ex

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
            self.balances.to_csv('balances.csv', index=False)

        # update positions
        async with self.positions_lock:
            for position in info['positions']:
                if position['symbol'] not in self.klines['symbol'].drop_duplicates().to_list():
                    continue
                primary_keys = ['symbol']
                side = position['ps']

                try:
                    last_price = await self.last_price(market, position['symbol'])
                    entry_price = float(position['entryPrice'])
                    pnl = float(position['unrealizedProfit'])
                    if (entry_price > last_price) == (pnl > 0):
                        side = 'SHORT'
                    elif (entry_price < last_price) == (pnl > 0):
                        side = 'LONG'
                except IndexError as ex:
                    logger.warning(f'update_position warning: {ex}')

                row = dict(
                    market='um-futures-cross',
                    symbol=position['symbol'],
                    amount=Decimal(position['positionAmt']),
                    entry_price=Decimal(position['entryPrice']),
                    pnl=float(position['unrealizedProfit']),
                    side=side,
                )
                self.positions = update_or_insert(
                    self.positions, row, primary_keys)
            self.positions.to_csv('positions.csv', index=False)

    async def on_tick(self, strategy: Strategy, klines: pd.DataFrame, is_kline_closed: bool):
        """ Not implemented stategy handler

        Args:
            strategy (Strategy): a strategy object
            klines (pd.DataFrame): klines df of strategy symbol
            is_kline_closed (bool): True, if kline is closed
        """
        raise NotImplementedError

    async def download_klines(self, strategy: Strategy) -> None:
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
        primary_keys = ['market', 'symbol', 'tf', 'open_time']
        async with self.klines_lock:
            self.klines = update_or_insert(self.klines, df, primary_keys)

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

    async def run_strategy(self, strategy: Strategy) -> None:
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

    async def run(self) -> None:
        """The main runbot method"""
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self.stop_handler, loop, self)
        loop.add_signal_handler(signal.SIGTSTP, self.stop_handler, loop, self)
        loop.add_signal_handler(signal.SIGTERM, self.stop_handler, loop, self)

        # update accounts info
        await self.update_accaunt_info(market='um-futures-cross')

        tasks = []
        # update markets info
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
