import asyncio
import pytest
from bot.base_bot import (
    invert_side,
)

def test_invert_side():
    assert invert_side('BUY') == 'SELL'
    assert invert_side('SELL') == 'BUY'