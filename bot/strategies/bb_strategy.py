from ..base_bot import BaseBot
from ..data_class import Strategy
from .preprocessing import calculate_indicators
import pandas as pd
import logging

logger = logging.getLogger(__name__)


async def on_tick(bot: BaseBot, strategy: Strategy, klines:pd.DataFrame, is_kline_closed: bool) -> None:
    def log_info(message:str):
        logger.info(f'{strategy.name}_{strategy.params.bb_period}_{strategy.params.bb_dev}: {message}')

    if not is_kline_closed:
        return
    
    # calculate indicators
    indicators = dict(
        bb=dict(period=strategy.params.bb_period, deviation=strategy.params.bb_dev)
    )
    df = calculate_indicators(klines, indicators)

    # log_info(f"bb_lower = {df.iloc[-1]['bb_lower']}")
    # log_info(f'length df={len(df)}')