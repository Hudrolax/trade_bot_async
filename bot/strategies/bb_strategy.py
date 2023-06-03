from ..base_bot import BaseBot, calculate_pnl
from ..data_class import Strategy
from .preprocessing import calculate_indicators
import pandas as pd
import logging
from typing import Any
import asyncio

logger = logging.getLogger(__name__)


async def on_tick(bot: BaseBot, strategy: Strategy, klines: pd.DataFrame, is_kline_closed: bool) -> None:
    def log_info(message: Any):
        logger.info(
            f'{strategy.name}_{strategy.symbol}_{strategy.params.bb_period}_{strategy.params.bb_dev}: {message}')

    if not is_kline_closed:
        return

    # calculate indicators
    indicators = dict(
        bb=dict(period=strategy.params.bb_period,
                deviation=strategy.params.bb_dev)
    )
    df = calculate_indicators(klines, indicators)

    # get info
    quote_asset = await bot.get_quote_asset(strategy)
    price = await bot.last_price(strategy.market, strategy.symbol)
    tick = df.iloc[-1]
    await asyncio.gather(
        bot.update_accaunt_info(strategy.market),
        # bot.update_open_orders(strategy),
    )
    positions: pd.DataFrame = await bot.get_strategy_positions(strategy)

    # cancel openned orders
    orders: pd.DataFrame = await bot.get_open_orders(strategy)
    for i, row in orders.iterrows():
        if await bot.cancel_order(strategy, row['id']):
            log_info(f"order {row['side']} {row['price']} was canceled.")

    # close positions
    for i, row in positions.iterrows():
        if (row['side'] == 'BUY' and price > tick['bb_middle']) \
                or (row['side'] == 'SELL' and price < tick['bb_middle']):
            if await bot.close_position(row, price):
                pnl = calculate_pnl(row['entry_price'], price, row['amount'])
                log_info(f"Try to close {row['side']} position at {price}. PNL {round(pnl, 2)}")

    # open a new order
    positions: pd.DataFrame = await bot.get_strategy_positions(strategy)
    balance = await bot.get_balance(strategy)

    if len(positions) == 0:
        if price < tick['bb_lower']:
        # BUY
            quantity = await bot.prepare_quantity(
                float(balance['ab']) * strategy.params.risk / 100,
                strategy,
                price,
            )
            if await bot.open_order(strategy, 'BUY', quantity, price):
                log_info(f'BUY on {price} ({quantity}) {quote_asset}')
        elif price > tick['bb_upper']:
            # SELL
            quantity = await bot.prepare_quantity(
                float(balance['ab']) * strategy.params.risk / 100,
                strategy,
                price,
            )
            if await bot.open_order(strategy, 'SELL', quantity, price):
                log_info(f'SELL on {price} ({quantity}) {quote_asset}')
