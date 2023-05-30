import aiohttp
import asyncio
import json

async def get_klines(symbol='btcusdt', interval='1m'):
    while True:
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol.upper(),
                'interval': interval
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()  # проверка на ошибки HTTP
                    klines = await response.text()
                    print(f"{symbol.upper()} klines: {json.loads(klines)}")
                    await asyncio.sleep(60)  # запрос каждую минуту
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(60)  # ожидание перед повторной попыткой

async def main():
    await get_klines()

# Запускаем главную асинхронную задачу
asyncio.run(main())