import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os
import sys

LOG_FORMAT = '%(name)s (%(levelname)s) %(asctime)s: %(message)s'
LOG_LEVEL = logging.INFO

# Loading .env if run without Docker
API_KEY = os.getenv('BINANCE_API_KEY', None)
if API_KEY is None:
    load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY', None)
SECRET_KEY = os.getenv('BINANCE_API_SECRET', None)

if API_KEY is None or SECRET_KEY is None:
    print('BINANCE_API_KEY and BINANCE_API_SECRET must be defined in .env file')
    sys.exit(1)


DEBUG = os.getenv('DEBUG', False)

BASE_FUTURES_URL = 'https://fapi.binance.com'
WSS_FUTURES_URL = 'wss://fstream.binance.com'
FUTURES_ORDER_URL = "/fapi/v1/order"
FUTURES_CANCEL_ALL_ORDERS_URL = '/fapi/v1/allOpenOrders'
FUTURES_LISTEN_KEY_URL = '/fapi/v1/listenKey'
FUTURES_KLINES_URL = '/fapi/v1/klines'
FUTURES_ACCOUNT_INFO_URL = '/fapi/v2/account'
FUTURES_MARKET_INFO_URL = '/fapi/v1/exchangeInfo'

if DEBUG:
    LOG_LEVEL = logging.DEBUG

log_dir = 'logs'
log_file = 'bot_log.txt'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

handler = RotatingFileHandler(os.path.join(
    'logs', log_file), maxBytes=2097152, backupCount=5)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(LOG_FORMAT)
console_handler.setFormatter(formatter)
handler.setFormatter(formatter)
logging.basicConfig(level=LOG_LEVEL, handlers=[handler, console_handler])