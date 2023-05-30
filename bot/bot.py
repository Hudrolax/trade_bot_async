from .base_bot import BaseBot
from .data_class import Strategy
from pandas import DataFrame
from .strategies.bb_strategy import on_tick as bb_on_tick
import logging

logger = logging.getLogger(__name__)

strategy_handlers = dict(
    bb=bb_on_tick,
)

class Bot(BaseBot):
    async def on_tick(self, strategy: Strategy, klines: DataFrame):
        try:
            await strategy_handlers[strategy.name](bot=self, strategy=strategy, klines=klines)
        except Exception as ex:
            logger.error(ex)
            raise ex