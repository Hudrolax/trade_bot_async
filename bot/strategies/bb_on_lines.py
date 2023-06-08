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
    sum_amount = positions['amount'].sum()

    max_quantity = await bot.prepare_quantity(
        float(balance['ab']) * risk / 100,
        strategy,
        price,
    )

    # defina orders plan
    orders_plan = dict()
    # middle order
    if abs(sum_amount) > 0:
        middle_price = await bot.prepare_price(tick['bb_middle'], strategy)
        quantity = await bot.prepare_quantity(
            abs(sum_amount),
            strategy,
            middle_price,
        )
        side = 'BUY' if sum_amount < 0 else 'SELL'
        orders_plan['middle'] = dict(
            price=middle_price, quantity=quantity, side=side)

    # lower/upper
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
        orders_plan['upper'] = dict(
            price=upper_price, quantity=quantity_sell, side='SELL')
        orders_plan['lower'] = dict(
            price=lower_price, quantity=quantity_buy, side='BUY')

    orders: pd.DataFrame = await bot.get_open_orders(strategy)

    tasks = []
    def handle_order(tasks: list,o_name:str, row: pd.Series):
        if o_name in orders_plan.keys():
            # modify, is needed
            price_diff = float(row['price']) - float(orders_plan[o_name]['price'])
            percent_diff = float(price_diff) / float(orders_plan[o_name]['price'])
            if percent_diff > 0.007 \
                or row['quantity'] != orders_plan[o_name]['quantity']:
                print(f'diff {percent_diff}')
                print(f"row quantity {row['quantity']}, plan quantity {orders_plan[o_name]['quantity']}")

                tasks.append(asyncio.create_task(
                    bot.modify_order(
                        order_id=row['id'],
                        strategy=strategy,
                        origClientOrderId=row['client_id'],
                        **orders_plan[o_name],
                    )
                ))
                log_info(f"{o_name} order {row['side']} id {row['id']} is modified.")
            else:
                log_info(f"Order {o_name} {row['side']} might be stay on the place.")
        else: # cancel the order
            tasks.append(asyncio.create_task(
                bot.cancel_order(strategy, row['id'])))
            log_info(
                f'Cancel the {o_name} {row["side"]} order with id {row["id"]}')
        # delete from plan (not open a new one)
        del orders_plan[o_name]

    for _, row in orders.iterrows():
        if '_close' in row['client_id']:
            handle_order(tasks, 'middle', row)
        elif '_upper' in row['client_id']:
            handle_order(tasks, 'upper', row)
        elif '_lower' in row['client_id']:
            handle_order(tasks, 'lower', row)
        else:
            tasks.append(asyncio.create_task(
                bot.cancel_order(strategy, row['id'])))
            log_info(
                f'Cancel unknown {row["side"]} order with id {row["client_id"]}')

    await asyncio.gather(*tasks)

    # open new orders
    tasks = []
    for key in orders_plan.keys():
        if key == 'upper':
            tasks.append(asyncio.create_task(
                bot.open_order(strategy=strategy, **orders_plan[key], newClientOrderId=f'{secrets.token_urlsafe(36)[:25]}_upper')
            ))
            log_info(f"Open a new upper order {orders_plan[key]['side']}")
        if key == 'lower':
            tasks.append(asyncio.create_task(
                bot.open_order(strategy=strategy, **orders_plan[key], newClientOrderId=f'{secrets.token_urlsafe(36)[:25]}_lower')
            ))
            log_info(f"Open a new lower order {orders_plan[key]['side']}")
        elif key == 'middle':
            tasks.append(asyncio.create_task(
                bot.open_order(strategy=strategy, **orders_plan[key], newClientOrderId=f'{secrets.token_urlsafe(36)[:25]}_close')
            ))
            log_info(f"Open a new middle order {orders_plan[key]['side']}")