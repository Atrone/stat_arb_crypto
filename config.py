import ccxt

# Configuration
EXCHANGES = ['vertex']  # Add more exchanges as needed
TIMEFRAME = '1h'  # Timeframe for OHLCV data
ARBITRAGE_THRESHOLD = 0  # Minimum arbitrage threshold (0.2%)
TRADE_SIZE = 100  # Example trade size, adjust as needed
CACHE_TTL = 300  # Example Time-To-Live value
IS_STAT = True
STAT_SYMBOL = 'BTC/USDC:USDC'

# Initialize CCXT exchange instances with API keys if necessary
EXCHANGE_INSTANCES = {exchange: getattr(ccxt, exchange)() for exchange in EXCHANGES}
