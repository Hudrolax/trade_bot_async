import traceback
import asyncio
import signal
import pandas as pd
import numpy as np
import logging
from bot.binance_um_futures import (
    wss_klines,
    get_klines,
    run_user_data_stream,
    get_account_info,
    get_market_info,
    get_open_orders,
    open_order,
    cancel_order,
    modify_order,
)
from bot.data_class import Strategy
from bot.errors import OpenOrderError, CloseOrderError, TickHandleError
from decimal import Decimal
from .strategies.preprocessing import float_to_decimal, count_significant_digits

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


def invert_side(side: str) -> str:
    if side == 'BUY':
        return 'SELL'
    elif side == 'SELL':
        return 'BUY'
    else:
        raise ValueError


def calculate_pnl(entry_price, exit_price, amount, commission=0.004):
    return (float(exit_price) - float(entry_price)) * float(amount) * (1 - commission)


def update_or_insert(df1: pd.DataFrame, new_row: dict | pd.DataFrame, keys: list) -> pd.DataFrame:
    """The function makes insert or update a dataframe

    Args:
        df1 (pd.DataFrame): the dataframe
        new_row (dict, pd.DataFrame): new row in dict format or one-row dataframe
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
    # change to decimal types
    for col in ['open', 'high', 'low', 'close', 'vol', 'qa_vol']:
        if col in df.columns and df[col].dtype != float:
            df[col] = df[col].astype(float)

    # change int types
    for col in ['trades']:
        if col in df.columns and df[col].dtype != int:
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
        self.klines = change_type_df(self.klines)

        self.balances: pd.DataFrame = pd.DataFrame([], columns=balances_cols)
        self.positions: pd.DataFrame = pd.DataFrame([], columns=positions_cols)
        self.orders: pd.DataFrame = pd.DataFrame([], columns=orders_cols)
        self.market_info: pd.DataFrame = pd.DataFrame(
            [], columns=market_info_cols)
        logger.info('****** Trade BOT ******')

    @staticmethod
    def error_handler(func):
        async def wrapper(self, *args, **kwargs):
            while not self.stop.is_set():
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    error_message = f"Exception occurred: {type(e).__name__}, {e.args}\n"
                    error_message += traceback.format_exc()
                    logger.critical(error_message)
                    await asyncio.sleep(3000)
        return wrapper

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
    
    async def prepare_price(self, price: int | float, strategy: Strategy) -> Decimal:
        async with self.market_lock:
            mask = (self.market_info['market'] == strategy.market) & (
                self.market_info['symbol'] == strategy.symbol)
            market_info = self.market_info[mask].iloc[0]
            price_precision = count_significant_digits(market_info['tickSize'])
            return float_to_decimal(price, price_precision)

    async def prepare_quantity(self, value: int | float | Decimal, strategy: Strategy, price: Decimal):
        async with self.market_lock:
            mask = (self.market_info['market'] == strategy.market) & (
                self.market_info['symbol'] == strategy.symbol)
            market_info = self.market_info[mask].iloc[0]
            step = count_significant_digits(market_info['stepSize'])
            min_notional = market_info['minNotional']

        quantity = float_to_decimal(float(value), step)
        min_notional_in_base_asset = float_to_decimal(
            float(min_notional) / float(price), step)
        if quantity < min_notional_in_base_asset:
            quantity = min_notional_in_base_asset
        return quantity

    async def get_quote_asset(self, strategy: Strategy) -> str:
        """Function returns quote asset"""
        async with self.market_lock:
            mask = (self.market_info['market'] == strategy.market) & (
                self.market_info['symbol'] == strategy.symbol)
            market_info = self.market_info[mask]
            if len(market_info) == 0:
                raise ValueError(
                    f'Have no market info for market/symbol {strategy.market}/{strategy.symbol}')
            market_info = market_info.iloc[0]
            return market_info['quoteAsset']

    async def open_order(
        self,
        strategy: Strategy,
        side: str,
        quantity: Decimal,
        price: Decimal | float,
        order_type: str = 'LIMIT',
        newClientOrderId: str = '',
    ) -> bool:
        """The function opens a new one order."""
        try:
            if isinstance(price, float):
                price = await self.prepare_price(price, strategy)
            await open_order(strategy.symbol, side, quantity, price, order_type, newClientOrderId)
        except OpenOrderError:
            return False

        return True

    async def modify_order(
        self,
        order_id: int,
        strategy: Strategy,
        side: str,
        quantity: Decimal,
        price: Decimal | float,
        origClientOrderId: str = '',
    ) -> bool:
        """The function midify an exists order."""
        try:
            if isinstance(price, float):
                price = await self.prepare_price(price, strategy)
            await modify_order(order_id, strategy.symbol, side, quantity, price, origClientOrderId)
        except OpenOrderError:
            return False

        return True

    async def cancel_order(
        self,
        strategy: Strategy,
        order_id: int,
    ) -> bool:
        """The function cancels an order."""
        try:
            await cancel_order(strategy.symbol, order_id=order_id)
        except CloseOrderError:
            return False

        return True

    async def close_position(self, position: pd.Series, price: Decimal | None = None) -> bool:
        """Close a position. If price is None - close by market price

        Args:
            position (pd.Series): series from self.positions df
            price (Decimal | None, optional): close price. Defaults to None.

        Returns:
            bool: True if closed
        """
        try:
            kwargs = dict(
                price=price,
                order_type='LIMIT',
            )
            if price is None:
                del kwargs['price']
                kwargs['order_type'] = 'MARKET'

            await open_order(
                symbol=position['symbol'],
                side=invert_side(position['side']),
                quantity=position['amount'],
                **kwargs
            )
        except OpenOrderError:
            return False
        return True

    async def get_strategy_positions(self, strategy: Strategy) -> pd.DataFrame:
        """The function gets positions for a certain strategy from dataframe"""
        async with self.positions_lock:
            mask = (self.positions['market'] == strategy.market) & (
                self.positions['symbol'] == strategy.symbol)
            return self.positions[mask].copy()

    async def get_open_orders(self, strategy: Strategy) -> pd.DataFrame:
        """The function return open orders for the strategy from dataframe"""
        async with self.orders_lock:
            mask = (self.orders['market'] == strategy.market) & (
                self.orders['symbol'] == strategy.symbol) & (
                (self.orders['status'] == 'NEW') | (self.orders['status'] == 'PARTIALLY_FILLED'))
            return self.orders[mask].copy()

    async def get_balance(self, strategy: Strategy) -> pd.Series:
        """The function gets balances Series from balances dataframe"""
        async with self.market_lock:
            mask = (self.market_info['market'] == strategy.market) & (
                self.market_info['symbol'] == strategy.symbol)
            quote_asset = self.market_info[mask].iloc[0]['quoteAsset']
        async with self.balances_lock:
            mask = (self.balances['market'] == strategy.market) & (
                self.balances['asset'] == quote_asset)
            balances = self.balances[mask].iloc[0]
        return balances

    @error_handler
    async def update_open_orders(self, strategy: Strategy) -> None:
        """The function gets open order from Binance and update the dataframe."""
        orders = await get_open_orders(strategy.symbol)
        primary_keys = ['market', 'symbol', 'id']
        async with self.orders_lock:
            # drop market/symbol rows first
            mask = (self.orders['market'] != strategy.market) & (
                self.orders['symbol'] != strategy.symbol)
            self.orders = self.orders[mask]

            # add new rows
            for order in orders:
                row = dict(
                    market=strategy.market,
                    symbol=strategy.symbol,
                    side=order['side'],
                    type=order['type'],
                    quantity=Decimal(order['origQty']),
                    price=float(order['price']),
                    average_price=float(order['avgPrice']),
                    status=order['status'],
                    id=order['orderId'],
                    client_id=order['clientOrderId'],
                    trade_time=pd.to_datetime(order['time'], unit='ms'),
                )
                self.orders = update_or_insert(self.orders, row, primary_keys)
            self.orders.to_csv('orders.csv', index=False)

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

    async def user_data_handler(self, data: dict, market: str):
        """The function updates account data by user data stream

        Args:
            data (dict): raw data from Binance
        """
        try:
            if data['e'] == "ACCOUNT_UPDATE":
                # update balances
                async with self.balances_lock:
                    primary_keys = ['market', 'asset']
                    for asset in data['a']['B']:
                        mask = (
                            self.balances['market'] == market) & (self.balances['asset'] == asset['a'])
                        wb = Decimal(asset['wb'])
                        cw = Decimal(asset['cw'])
                        old_ab = self.balances[mask].iloc[-1]['ab']
                        row = dict(
                            market=market,
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
                    primary_keys = ['market', 'symbol']
                    for position in data['a']['P']:
                        # update changes
                        amount = Decimal(position['pa'])
                        entry_price = float(position['ep'])
                        pnl = float(position['up'])
                        row = dict(
                            market=market,
                            symbol=position['s'],
                            amount=amount,
                            entry_price=entry_price,
                            pnl=pnl,
                            side='BUY' if amount >= 0 else 'SELL',
                        )
                        self.positions = update_or_insert(
                            self.positions, row, primary_keys)
                        # logger.info(f"positions updated for {position['s']}")

                    # del closed positions
                    indexes = self.positions[self.positions['amount'] == 0].index
                    self.positions = self.positions.drop(
                        indexes).reset_index(drop=True)

                    # logger.info(f"positions: {self.positions}")

                    self.positions.to_csv('positions.csv', index=False)

            elif data['e'] == 'ORDER_TRADE_UPDATE':
                order = data['o']
                row = dict(
                    market=market,
                    symbol=order['s'],
                    side=order['S'],
                    type=order['o'],
                    quantity=Decimal(order['q']),
                    price=float(order['p']),
                    average_price=float(order['ap']),
                    status=order['X'],
                    id=order['i'],
                    client_id=order['c'],
                    trade_time=pd.to_datetime(order['T'], unit='ms'),
                )
                primary_keys = ['market', 'symbol', 'id']
                async with self.orders_lock:
                    self.orders = update_or_insert(
                        self.orders, row, primary_keys)
                    # logger.info(f"order for {order['s']} was updated")
                    # logger.info(f'orders: {self.orders}')
                    self.orders.to_csv('orders.csv', index=False)
        except Exception as e:
            error_message = f"Exception occurred: {type(e).__name__}, {e.args}\n"
            error_message += traceback.format_exc()
            logger.critical(error_message)

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
                trades=int(kline['n']),
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
        except Exception as e:
            error_message = f"Exception occurred: {type(e).__name__}, {e.args}\n"
            error_message += traceback.format_exc()
            logger.critical(error_message)

    @error_handler
    async def update_accaunt_info(self, market: str) -> None:
        """The function update account info for a specific market.

        Args:
            market (str): market name
        """
        info: dict = await get_account_info()
        # update balances
        async with self.balances_lock:
            # clear balances
            mask = self.balances['market'] != market
            self.balances = self.balances[mask]
            # add new balances
            primary_keys = ['market', 'asset']
            for asset in info['assets']:
                row = dict(
                    market=market,
                    asset=asset['asset'],
                    wb=Decimal(asset['walletBalance']),
                    cw=Decimal(asset['crossWalletBalance']),
                    ab=Decimal(asset['availableBalance']),
                )
                self.balances = update_or_insert(
                    self.balances, row, primary_keys)
            self.balances.to_csv('balances.csv', index=False)

        # update positions
        async with self.positions_lock:
            # clear positions
            mask = self.positions['market'] != market
            self.positions = self.positions[mask]
            # add new positions
            for position in info['positions']:
                amount = float(position['positionAmt'])
                if float(position['entryPrice']) == 0 or abs(amount) == 0:
                    continue
                primary_keys = ['market', 'symbol']
                row = dict(
                    market=market,
                    symbol=position['symbol'],
                    amount=amount,
                    entry_price=float(position['entryPrice']),
                    pnl=float(position['unrealizedProfit']),
                    side='BUY' if amount >= 0 else 'SELL',
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

    @error_handler
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
        primary_keys = ['market', 'symbol', 'tf', 'open_time']
        async with self.klines_lock:
            self.klines = update_or_insert(self.klines, df, primary_keys)

    @error_handler
    async def get_market_info(self, market: str) -> None:
        if market == 'um-futures-cross' or market == 'um-futures':
            raw_data = await get_market_info(self.stop)
        else:
            raise ValueError(f'Wrong market name: {market}')

        async with self.market_lock:
            # clear an old market info
            mask = self.market_info['market'] != market
            self.market_info = self.market_info[mask]
            # add a new market info
            for symbol in raw_data['symbols']:
                row = dict(
                    market=market,
                    symbol=symbol['symbol'],
                    status=symbol['status'],
                    baseAsset=symbol['baseAsset'],
                    quoteAsset=symbol['quoteAsset'],
                    pricePrecision=symbol['pricePrecision'],
                    quantityPrecision=symbol['quantityPrecision'],
                    baseAssetPrecision=symbol['baseAssetPrecision'],
                    quotePrecision=symbol['quotePrecision'],
                )
                for filtr in symbol['filters']:
                    if filtr['filterType'] == 'PRICE_FILTER':
                        row['tickSize'] = filtr['tickSize']
                    elif filtr['filterType'] == 'LOT_SIZE':
                        row['maxQty'] = Decimal(filtr['maxQty'])
                        row['minQty'] = Decimal(filtr['minQty'])
                        row['stepSize'] = Decimal(filtr['stepSize'])
                    elif filtr['filterType'] == 'MIN_NOTIONAL':
                        row['minNotional'] = Decimal(filtr['notional'])

                self.market_info = update_or_insert(
                    self.market_info, row, ['market', 'symbol'])
                self.market_info.to_csv('market_info.csv', index=False)

    async def update_market_info(self, market: str) -> None:
        """The function updates market info"""
        while not self.stop.is_set():
            await self.get_market_info(market)
            await asyncio.sleep(60 * 60)  # one hour

    async def run_strategy(self, strategy: Strategy) -> None:
        """Function runs an infinity loop for every strategy"""
        logger.info(
            f"Run strategy {strategy.name} for {strategy.symbol}_{strategy.tf} on {strategy.market}")
        try:
            # update open orders
            await self.update_open_orders(strategy)

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
            run_user_data_stream(self.user_data_handler, self.stop, 'um-futures-cross')))

        # run strategies
        for strategy in self.strategies:
            tasks.append(asyncio.create_task(self.run_strategy(strategy)))

        await self.stop.wait()
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks)
