import asyncio
import aiohttp
import websockets
import json
from config import (
    WSS_FUTURES_URL,
    BASE_FUTURES_URL,
    API_KEY,
    SECRET_KEY,
    FUTURES_ORDER_URL,
    FUTURES_CANCEL_ALL_ORDERS_URL,
    FUTURES_KLINES_URL,
    FUTURES_LISTEN_KEY_URL,
    FUTURES_ACCOUNT_INFO_URL,
    FUTURES_MARKET_INFO_URL,
    FUTURES_OPEN_ORDERS_URL,
)
from .data_class import Strategy
import logging
import hashlib
import hmac
import time
import urllib.parse
from decimal import Decimal
from .errors import OpenOrderError, CloseOrderError, TickHandleError
from typing import Callable, Coroutine

logger = logging.getLogger(__name__)


async def unauthorizrd_request(
    endpoint: str,
    http_method: str,
    params: dict,
    logger: logging.Logger,
    stop: asyncio.Event,
) -> dict:
    """The function makes unauthorized request

    Args:
        endpoint (str): endpoint to request
        http_method (str): request method (like get, post)
        params (dict): params for request
        logger (logging.Logger): actual logger
        stop (asyncio.Event): stop Event for stopping attempts

    Returns:
        dict: parsed JSON response
    """
    for _ in range(5):  # do 5 attempts
        if stop.is_set():
            raise asyncio.CancelledError

        try:
            url = BASE_FUTURES_URL + endpoint
            async with aiohttp.ClientSession() as session:
                async with session.request(http_method, url, params=params) as response:
                    response.raise_for_status()  # проверка на ошибки HTTP
                    text = await response.text()
                    logger.debug(f"raw response: {text}")
                    return json.loads(text)

        except aiohttp.ClientError as e:
            logger.warning(f"An error occurred: {e}")
            await asyncio.sleep(2)  # pause before next attemption
    raise aiohttp.ClientError("Connection error.")


async def authorized_request(
    endpoint: str,
    http_method: str,
    params: dict,
    ErrorClass: Callable,
    logger: logging.Logger,
) -> dict | list:
    """The function makes authorized request to API (trades)

    Args:
        endpoint (str): request endpoint
        http_method (str): HTTP method (post, delete)
        params (dict): params for request
        ErrorClass (Callable): Error class for raising an error
        logger (logging.Logger): logger object for correct logging

    Returns:
        dict: parsed JSON response from the broker
    """
    url = BASE_FUTURES_URL + endpoint

    # make the signature
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(str(SECRET_KEY).encode(),
                         query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature

    headers = {
        "X-MBX-APIKEY": API_KEY
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(http_method, url, params=params, headers=headers) as response:
                response.raise_for_status()  # проверка на ошибки HTTP
                text = await response.text()
                logger.debug(f"response: {text}")
                return json.loads(text)
    except aiohttp.ClientResponseError as e:
        logger.error(
            f"An error occurred: {str(e)}, status code: {e.status}, message: {e.message}")
        raise ErrorClass(f'{e}')
    except aiohttp.ClientError as e:
        logger.warning(f"An error occurred: {str(e)}")
        raise ErrorClass(f'{e}')
    except Exception as e:
        logger.error(e)
        raise ErrorClass(f'unexpected error {e}')


async def get_account_info(**kwargs) -> dict:
    """The function gets account info

    Returns:
        dict: parsed JSON data
    """
    logger = logging.getLogger('account_info')
    params = {
        "timestamp": int(time.time() * 1000),  # Timestamp in milliseconds
        **kwargs
    }
    response = authorized_request(
        FUTURES_ACCOUNT_INFO_URL, 'get', params, TickHandleError, logger)
    if not isinstance(response, dict):
        raise TypeError
    return response


async def open_order(symbol: str, side: str, quantity: Decimal, price: Decimal, order_type: str = 'LIMIT', **kwargs) -> dict:
    """The function opens an order

    Args:
        symbol (str): Symbol, like BTCUSDT
        side (str): BUY or SELL
        quantity (Decimal): Quantity on quote asset. For BTCUSDT it's amount of BTC
        price (Decimal): price in base asset. For BTCUSDT it's price in USDT.
        order_type (str, optional): ORDER type. See types in official Binance documentation. Defaults to 'LIMIT'.
        **kwargs: option kwargs. See official Binance documentation.

    Returns:
        dict: parsed JSON response
    """
    logger = logging.getLogger('open_order')
    # order parameters
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "timeInForce": "GTC",
        "quantity": str(quantity),
        "price": str(price),
        "timestamp": int(time.time() * 1000),  # Timestamp in milliseconds
        **kwargs
    }
    response = await authorized_request(FUTURES_ORDER_URL, 'post', params, OpenOrderError, logger)
    if not isinstance(response, dict):
        raise TypeError
    return response


async def cancel_order(symbol: str, order_id: int = -1, origClientOrderId: str = '', **kwargs) -> dict:
    """The function cancels an order (not position)

    Args:
        symbol (str): symbol like BTCUSDT
        order_id (int, optional): Order ID from Binance. Defaults to -1.
        origClientOrderId (str, optional): Client side order ID. Defaults to ''.
        **kwargs: option kwargs. See official Binance documentation.
        !!! order_id or origClientOrderId - one of them MUST be filled.

    Returns:
        dict: parsed JSON response
    """
    logger = logging.getLogger('cancel_order')
    # Параметры ордера
    if order_id == -1 and origClientOrderId == '':
        raise CloseOrderError('order_id or origClientOrderId must be sent.')

    params = {
        "symbol": symbol,
        "orderId": order_id,
        "timestamp": int(time.time() * 1000),  # Timestamp in milliseconds
        **kwargs
    }
    if order_id == -1:
        del params['orderId']
        params['origClientOrderId'] = origClientOrderId

    response = await authorized_request(FUTURES_ORDER_URL, 'delete', params, CloseOrderError, logger)
    if not isinstance(response, dict):
        raise TypeError
    return response


async def cancel_all_orders(symbol: str, **kwargs) -> bool:
    """The function cancels all opened orders (not positions)

    Args:
        symbol (str): symbol like BTCUSDT

    Returns:
        bool: True, if all orders was cancelled.
    """
    logger = logging.getLogger('cancel_all_orders')
    params = {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),  # Timestamp in milliseconds
        **kwargs
    }
    response = await authorized_request(FUTURES_CANCEL_ALL_ORDERS_URL, 'delete', params, CloseOrderError, logger)
    if not isinstance(response, dict):
        raise TypeError
    return response['code'] == 200


async def get_open_orders(symbol: str, **kwargs) -> list:
    logger = logging.getLogger('get_open_orders')
    params = {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),  # Timestamp in milliseconds
        **kwargs
    }
    response = await authorized_request(FUTURES_OPEN_ORDERS_URL, 'get', params, TickHandleError, logger)
    if not isinstance(response, list):
        raise TypeError
    return response


async def get_klines(symbol: str, interval: str, stop: asyncio.Event, limit: int = 500) -> dict:
    """The function gets klines history from Binance

    Args:
        symbol (str): symbol like BTCUSDT
        interval (str): interval (timeframe) like '15m'
        stop (asyncio.Event): stop event from the bot
        limit (int, optional): limit of klines. Defaults to 500.

    Returns:
        dict: parsed JSON response
    """

    logger = logging.getLogger('get_klines')
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit,
    }
    return await unauthorizrd_request(FUTURES_KLINES_URL, 'get', params, logger, stop)


async def get_market_info(stop: asyncio.Event) -> dict:
    """The function gets exchange info (symbols, limits)

    Args:
        stop (asyncio.Event): asyncio stop event

    Returns:
        dict: parsed JSON data
    """
    logger = logging.getLogger('get_market_info')
    params = {}
    return await unauthorizrd_request(FUTURES_MARKET_INFO_URL, 'get', params, logger, stop)


async def wss_klines(handler: Callable, strategy: Strategy, stop_event: asyncio.Event):
    """ The function runs websocket stream for receiving kline data every 250ms
    for one strategy.

    Args:
        handler (Callable): handler func for handle the data
        strategy (Strategy): strategy (symbol/timeframe) for start stream
        stop_event (asyncio.Event): stop event
    """
    logger = logging.getLogger('wss_klines')
    symbol = strategy.symbol.lower()
    interval = strategy.tf
    url = f'{WSS_FUTURES_URL}/ws/{symbol}@kline_{interval}'
    while not stop_event.is_set():
        try:
            async with websockets.connect(url) as ws:
                while not stop_event.is_set():
                    kline = await ws.recv()
                    logger.debug(f'raw kline: {kline}')
                    kline_dict = json.loads(kline)
                    if kline_dict['e'] == 'kline':
                        await handler(strategy=strategy, data=kline_dict)
                    else:
                        raise ValueError(
                            "Kline response doesn't contain klines")
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
            logger.warning("Connection closed, retrying...")
            await asyncio.sleep(1)  # waiting before reconnect
        except asyncio.CancelledError as e:
            raise e
        except Exception as e:
            logger.critical(f"An unexpected error from strategy: {e}")
            await asyncio.sleep(1)  # waiting before reconnect


async def get_listen_key(session: aiohttp.ClientSession) -> str:
    """The function gets listen key for start user-data stream.

    Args:
        session (aiohttp.ClientSession): Session object

    Returns:
        str: the listen key
    """
    headers = {
        "X-MBX-APIKEY": API_KEY,
    }
    url = BASE_FUTURES_URL + FUTURES_LISTEN_KEY_URL
    async with session.post(url, headers=headers) as response:
        data = await response.json()
        return data['listenKey']


async def keepalive_listen_key(session: aiohttp.ClientSession, listen_key: str):
    """The function for keep alive listen key

    Args:
        session (aiohttp.ClientSession): the session
        listen_key (str): exist listen key
    """
    headers = {
        "X-MBX-APIKEY": API_KEY,
    }
    url = BASE_FUTURES_URL + FUTURES_LISTEN_KEY_URL
    params = {
        "listenKey": listen_key
    }
    await session.put(url, params=params, headers=headers)


async def handle_stream(
    session: aiohttp.ClientSession,
    listen_key: str,
    handler: Callable,
    stop_event: asyncio.Event,
    logger: logging.Logger,
) -> None:
    """User data stream handler

    Args:
        session (aiohttp.ClientSession): active session
        listen_key (str): exist listen key
        handler (Callable): data handler
        stop_event (asyncio.Event): the stop event
        logger (logging.Logger): actual logger
    """
    while not stop_event.is_set():
        try:
            wss_url = f"{WSS_FUTURES_URL}/ws/{listen_key}"
            async with session.ws_connect(wss_url) as ws:
                async for msg in ws:
                    # Handle received message
                    logger.debug(msg)
                    await handler(msg.json())
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
            logger.warning(
                f"Stream encountered an error: {e}. Reconnecting...")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(
                f"Stream encountered an error: {e}.Reconnecting...")
            await asyncio.sleep(1)


async def run_user_data_stream(data_handler: Callable, stop_event: asyncio.Event):
    """The function runs user-data stream

    Args:
        data_handler (Callable): user-data handler
        stop_event (asyncio.Event): the stop event
    """
    logger = logging.getLogger('user_data_stream')
    async with aiohttp.ClientSession() as session:
        listen_key = await get_listen_key(session)
        logger.debug(f"Got listen key: {listen_key}")

        # Launch the stream handler
        handler = asyncio.create_task(
            handle_stream(session, listen_key, data_handler, stop_event, logger))

        while not stop_event.is_set():
            await asyncio.sleep(60*50)  # 50 minutes

            try:
                # Refresh listen key
                await keepalive_listen_key(session, listen_key)
                logger.debug("Refreshed listen key.")
            except Exception as e:
                logger.warning(
                    f"Failed to refresh listen key: {e}. Getting a new one...")
                listen_key = await get_listen_key(session)
                logger.debug(f"Got new listen key: {listen_key}")

                # Cancel the old handler and start a new one with the new listen key
                handler.cancel()
                await handler
                handler = asyncio.create_task(
                    handle_stream(session, listen_key, data_handler, stop_event, logger))
