from ..base_bot import BaseBot
from ..data_class import Strategy
import pandas as pd
import logging

logger = logging.getLogger(__name__)

async def on_tick(bot: BaseBot, strategy: Strategy, klines:pd.DataFrame) -> None:
    try:
        print(f'{strategy.name} on tick!!')
    except Exception as ex:
        logger.error(ex)
        raise ex