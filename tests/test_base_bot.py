import asyncio
import pytest
from decimal import Decimal
from bot.base_bot import (
    invert_side,
    update_or_insert,
    BaseBot,
)
from bot.strategies.preprocessing import count_significant_digits
import pandas as pd

def test_count_significant_digits() -> None:
    """Testing counting significant digits after the dot."""
    assert count_significant_digits(5) == 0
    assert count_significant_digits("5") == 0
    assert count_significant_digits(Decimal(5)) == 0
    assert count_significant_digits(Decimal("5")) == 0
    assert count_significant_digits(Decimal("5.0")) == 0
    assert count_significant_digits(Decimal(5.0)) == 0
    assert count_significant_digits(0.1 + 0.2) == 1
    assert count_significant_digits(Decimal(0.1 + 0.2)) == 1
    assert count_significant_digits(0.00001) == 5
    assert count_significant_digits("0.00001") == 5
    assert count_significant_digits(Decimal("0.00001")) == 5
    assert count_significant_digits(Decimal(0.00001)) == 5
    assert count_significant_digits(0.0000100) == 5
    assert count_significant_digits("0.0000100") == 5
    assert count_significant_digits(Decimal("0.0000100")) == 5
    assert count_significant_digits(Decimal(0.0000100)) == 5


def test_invert_side():
    assert invert_side('BUY') == 'SELL'
    assert invert_side('SELL') == 'BUY'

def test_update_or_insert():
    df1 = pd.DataFrame(dict(
        a=[1, 2, 3],
        b=[2, 3, 4],
    ))
    df2 = pd.DataFrame(dict(
        a=[1, 2, 3, 4],
        b=[10, 3, 4, 5],
    ))
    df_merged = update_or_insert(df1, df2, keys=['a'])
    assert df_merged.loc[0, 'b'] == 10
    assert df_merged.loc[3, 'b'] == 5

@pytest.fixture(scope='module')
def bot():
    strategies = [
        dict(
            name='bb',
            market='um-futures-cross',
            symbol='SUIUSDT',
            tf='15m',
            window=25,
            params=dict(
                bb_period=20,
                bb_dev=2.0,
                risk=30,
            )
        ),
    ]
    return BaseBot(strategies)

@pytest.mark.asyncio
async def test_bot(bot):
    strategy = bot.strategies[0]
    # get data
    await asyncio.gather(
        bot.update_accaunt_info(market = 'um-futures-cross'),
        bot.download_klines(strategy),
        bot.get_market_info(strategy.market),
        bot.update_open_orders(strategy)
    )
    assert len(bot.market_info) > 0

    # market info tests
    quote_asset = await bot.get_quote_asset(strategy)
    assert quote_asset == 'USDT'
    assert await bot.prepare_price(0.777777000000003, strategy) == Decimal('0.7778')
    assert await bot.prepare_quantity(54.3000000000000003, strategy, Decimal('0.815400')) == Decimal('54.3')


    # balances tests
    assert len(bot.balances) > 0
    assert isinstance(bot.balances[bot.balances['asset'] == 'USDT'].iloc[-1]['ab'], Decimal) 
    balance = await bot.get_balance(strategy)
    assert isinstance(balance, pd.Series)
    assert isinstance(balance['ab'], Decimal)

    # orders tests
    orders = await bot.get_open_orders(strategy)
    assert isinstance(orders, pd.DataFrame)
    assert 'market' in orders.columns
    assert 'symbol' in orders.columns

    # positions tests
    positions = await bot.get_strategy_positions(strategy)
    assert isinstance(positions, pd.DataFrame)
    assert 'amount' in positions.columns

    # klines tests
    assert len(bot.klines) > 0
    assert bot.klines['close'].dtype == float
    assert bot.klines['close'].mean() / 2 > 0
    last_price = await bot.last_price(strategy.market, strategy.symbol)
    assert isinstance(last_price, float)
    mask = (bot.klines['market'] == strategy.market) & (bot.klines['symbol'] == strategy.symbol)
    lp = bot.klines[mask].iloc[-1]['close']
    assert last_price == lp

