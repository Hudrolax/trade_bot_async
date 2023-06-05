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
            symbol='SUIUSDT',
            tf='15m',
            window=100,
            params=dict(
                bb_period=20,
                bb_dev=2.1,
                risk=30,
            )
        ),
        dict(
            name='bb',
            market='um-futures-cross',
            symbol='USDCUSDT',
            tf='15m',
            window=100,
            params=dict(
                bb_period=20,
                bb_dev=2.4,
                risk=50,
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
