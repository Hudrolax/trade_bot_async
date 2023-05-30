import aiohttp
import asyncio
import websockets
import json
import hmac
import hashlib
import time

from dotenv import load_dotenv
import os

# Загрузить переменные окружения из файла .env
load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_API_SECRET')
BASE_URL = 'https://fapi.binance.com'

def get_signature(query_string, secret_key):
    return hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

async def get_listen_key():
    url = f"{BASE_URL}/fapi/v1/listenKey"
    timestamp = int(time.time() * 1000)
    headers = {
        'X-MBX-APIKEY': API_KEY
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers) as response:
            response.raise_for_status()  
            data = await response.json()
            return data['listenKey']

async def user_data_stream():
    listen_key = await get_listen_key()
    print(f'listen key {listen_key}')
    url = f"wss://fstream.binance.com/ws/{listen_key}"

    while True:
        try:
            async with websockets.connect(url) as ws:
                print('connected to socket')
                while True:
                    data = await ws.recv()
                    data_dict = json.loads(data)
                    print(data_dict)

                    # if data_dict['e'] == 'ORDER_TRADE_UPDATE':
                    #     print("Order update:", data_dict)

        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
            print("Connection closed, retrying...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            await asyncio.sleep(1)

async def main():
    await user_data_stream()

# Запускаем главную асинхронную задачу
asyncio.run(main())
