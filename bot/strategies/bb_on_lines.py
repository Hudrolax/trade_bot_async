from ..base_bot import BaseBot, calculate_pnl
from ..data_class import Strategy
from .preprocessing import calculate_indicators
import pandas as pd
import logging
from typing import Any
import asyncio
from decimal import Decimal
import secrets

logger = logging.getLogger(__name__)


async def on_tick(bot: BaseBot, strategy: Strategy, klines: pd.DataFrame, is_kline_closed: bool) -> None:
    if not is_kline_closed:
        return

    name = strategy.name
    symbol = strategy.symbol
    bb_period = strategy.params.bb_period
    bb_dev = strategy.params.bb_dev
    market = strategy.market
    risk = strategy.params.risk

    def log_info(message: Any):
        logger.info(
            f'{name}_{symbol}_{bb_period}_{bb_dev}: {message}')

    # calculate indicators
    indicators = dict(
        bb=dict(period=bb_period,
                deviation=bb_dev)
    )
    df = calculate_indicators(klines, indicators)

    # get info
    quote_asset = await bot.get_quote_asset(strategy)
    price_float: float = await bot.last_price(market, symbol)
    price: Decimal = await bot.prepare_price(price_float, strategy)
    tick = df.iloc[-1]
    await asyncio.gather(
        # for update available balance (with openned positions on cross account)
        bot.update_accaunt_info(market),
        # bot.update_open_orders(strategy),
    )
    positions: pd.DataFrame = await bot.get_strategy_positions(strategy)
    balance = await bot.get_balance(strategy)

    max_quantity = await bot.prepare_quantity(
        float(balance['ab']) * risk / 100,
        strategy,
        price,
    )

    # cancel openned orders
    orders: pd.DataFrame = await bot.get_open_orders(strategy)

    tasks = []
    for _, row in orders.iterrows():
        if '_close' in row['client_id']:
            log_info(f"Cancel the middle-order with id {row['id']}")
        elif '_new' in row['client_id'] and row['side'] == 'BUY':
            log_info(f'Cancel the BUY order with id {row["id"]}')
        elif '_new' in row['client_id'] and row['side'] == 'SELL':
            log_info(f'Cancel the SELL order with id {row["id"]}')
        tasks.append(asyncio.create_task(
            bot.cancel_order(strategy, row['id'])))

    # open orders
    positions: pd.DataFrame = await bot.get_strategy_positions(strategy)
    orders: pd.DataFrame = await bot.get_open_orders(strategy)
    sum_amount = positions['amount'].sum()

    if abs(sum_amount) > 0:
        # open order for closing exists position
        middle_price = await bot.prepare_price(tick['bb_middle'], strategy)
        quantity = await bot.prepare_quantity(
            abs(sum_amount),
            strategy,
            middle_price,
        )
        side = 'BUY' if sum_amount < 0 else 'SELL'
        tasks.append(asyncio.create_task(
            bot.open_order(strategy, side, quantity, middle_price,
                           newClientOrderId=f'{secrets.token_urlsafe(36)[:25]}_close'),
        ))
        log_info(f'position amount: {sum_amount}')
        log_info(
            f'open a middle-order to close the position: side {side} amount {quantity} price {middle_price}')

    if abs(sum_amount) < max_quantity:
        lower_price = await bot.prepare_price(tick['bb_lower'], strategy)
        upper_price = await bot.prepare_price(tick['bb_upper'], strategy)
        amount = float(max_quantity) - sum_amount
        if amount > max_quantity:
            amount = max_quantity

        quantity_buy = await bot.prepare_quantity(
            float(amount),
            strategy,
            lower_price,
        )

        quantity_sell = await bot.prepare_quantity(
            float(amount),
            strategy,
            upper_price,
        )

        result = await asyncio.gather(
            bot.open_order(strategy, 'BUY', quantity_buy, lower_price,
                           newClientOrderId=f'{secrets.token_urlsafe(36)[:25]}_new'),
            bot.open_order(strategy, 'SELL', quantity_sell,
                           upper_price, newClientOrderId=f'{secrets.token_urlsafe(36)[:25]}_new'),
        )
        if result[0]:
            log_info(f"BUY on {lower_price} ({quantity_buy}) {quote_asset}")
        if result[1]:
            log_info(f'SELL on {upper_price} ({quantity_sell}) {quote_asset}')
