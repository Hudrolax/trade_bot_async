import config
import asyncio
from bot.bot import Bot
import logging

logger = logging.getLogger('main')

async def main():
    strategies = [
        # dict(
        #     name='bb',
        #     market='um-futures-cross',
        #     symbol='BTCUSDT',
        #     tf='1m',
        #     window=100,
        #     params=dict(
        #         bb_period=20,
        #         bb_dev=2.0,
        #         risk=5,
        #     )
        # ),
        dict(
            name='bb',
            market='um-futures-cross',
            symbol='DOGEUSDT',
            tf='15m',
            window=100,
            params=dict(
                bb_period=20,
                bb_dev=2.0,
                risk=20,
            )
        ),
    ]

    bot = Bot(strategies=strategies)
    try:
        await bot.run()
    except asyncio.exceptions.CancelledError:
        logger.info('Exit')

if __name__ == '__main__':
    asyncio.run(main())
