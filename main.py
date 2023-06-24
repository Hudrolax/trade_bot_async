import config
import asyncio
from bot.bot import Bot
import logging

logger = logging.getLogger('main')

async def main():
    strategies = [
        dict(
            name='bb',
            market='um-futures-cross',
            symbol='COMBOUSDT',
            tf='15m',
            window=100,
            params=dict(
                bb_period=20,
                bb_dev=2,
                risk=30,
            )
        ),
        # dict(
        #     name='bb_on_lines',
        #     market='um-futures-cross',
        #     symbol='USDCUSDT',
        #     tf='15m',
        #     window=100,
        #     params=dict(
        #         bb_period=60,
        #         bb_dev=2.2,
        #         risk=1000,
        #     )
        # ),
    ]

    bot = Bot(strategies=strategies)
    try:
        await bot.run()
    except asyncio.exceptions.CancelledError:
        logger.info('Exit')

if __name__ == '__main__':
    asyncio.run(main())
