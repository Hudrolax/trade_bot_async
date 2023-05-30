import asyncio
import websockets
import json

async def bnb_klines(symbol='btcusdt', interval='1m'):
    url = f'wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}'
    while True:
        try:
            async with websockets.connect(url) as ws:
                while True:
                    kline = await ws.recv()  # получаем обновления
                    kline_dict = json.loads(kline)
                    print(f"{symbol.upper()} price update: {kline_dict['k']['c']}")
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
            print("Connection closed, retrying...")
            await asyncio.sleep(1)  # ожидание перед повторной попыткой подключения
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            await asyncio.sleep(1)  # ожидание перед повторной попыткой подключения

async def main():
    await bnb_klines()

# Запускаем главную асинхронную задачу
asyncio.run(main())