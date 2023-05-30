from bot.binance_um_futures import open_order, cancel_all_orders
import asyncio
from decimal import Decimal

# asyncio.run(open_order("DOGEUSDT", "SELL", quantity=Decimal('100'), price=Decimal('0.074')))
print(asyncio.run(cancel_all_orders(symbol='DOGEUSDT')))