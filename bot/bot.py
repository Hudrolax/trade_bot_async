from .base_bot import BaseBot
from .data_class import Strategy
from pandas import DataFrame
from .strategies.bb_strategy import on_tick as bb_on_tick
from .strategies.bb_on_lines import on_tick as bb_on_tick_on_lines
import logging

logger = logging.getLogger(__name__)

strategy_handlers = dict(
    bb=bb_on_tick,
    bb_on_lines=bb_on_tick_on_lines,
)


class Bot(BaseBot):
    async def on_tick(self, strategy: Strategy, klines: DataFrame, is_kline_closed: bool):
        try:
            await strategy_handlers[strategy.name](
                bot=self,
                strategy=strategy,
                klines=klines,
                is_kline_closed=is_kline_closed,
            )
        except Exception as ex:
            logger.error(ex)
            raise ex
