from ..base_bot import BaseBot
from ..data_class import Strategy
from .preprocessing import calculate_indicators
import pandas as pd
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def on_tick(bot: BaseBot, strategy: Strategy, klines: pd.DataFrame, is_kline_closed: bool) -> None:
    def log_info(message: Any):
        logger.info(
            f'{strategy.name}_{strategy.params.bb_period}_{strategy.params.bb_dev}: {message}')

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
    await bot.update_accaunt_info(strategy.market)
    positions: pd.DataFrame = await bot.get_strategy_positions(strategy)

    # close positions
    for i, row in positions.iterrows():
        if (row['side'] == 'BUY' and price > tick['bb_middle']) \
                or (row['side'] == 'SELL' and price < tick['bb_middle']):
            if await bot.close_position(row, price):
                log_info(f"position closed. PNL {row['pnl']}")

    # open new order
    positions: pd.DataFrame = await bot.get_strategy_positions(strategy)
    balance = await bot.get_balance(strategy)

    if len(positions) == 0:
        if price < tick['bb_lower']:
            quantity = await bot.prepare_quantity(
                float(balance['ab']) * strategy.params.risk / 100,
                strategy,
                price,
            )
            if await bot.open_order(strategy, 'BUY', quantity, price):
                log_info(f'BUY on {price} ({quantity}) {quote_asset}')
        elif price > tick['bb_upper']:
            quantity = await bot.prepare_quantity(
                float(balance['ab']) * strategy.params.risk / 100,
                strategy,
                price,
            )
            if await bot.open_order(strategy, 'SELL', quantity, price):
                log_info(f'BUY on {price} ({quantity}) {quote_asset}')
