from datetime import datetime, timedelta
import ccxt
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import time
import pandas as pd

class PriceMonitor:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.active_positions: Dict[str, Dict] = {}
        
    def fetch_minute_data(self, symbol: str, current_time: str, 
                              lookback_minutes: int = 60) -> List[Dict]:
        """
        Fetch minute-by-minute price data for a symbol
        
        :param symbol: Trading pair symbol
        :param current_time: Simulated current time being tested
        :param lookback_minutes: Number of historical minutes to fetch
        :return: List of OHLCV data
        """
        try:
            # Convert string to datetime if needed
            if isinstance(current_time, str):
                current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
            print(current_time)    
            since = self.exchange.parse8601(current_time.isoformat())
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                '1m', 
                since=since,
                limit=lookback_minutes
            )
            print(ohlcv)
            return ohlcv
        except Exception as e:
            logging.error(f"Error fetching minute data for {symbol} at {current_time}: {e}")
            return []

    def monitor_position(self, symbol: str, entry_price: float, position_type: str, 
                            stop_loss_pct: float, current_time: datetime) -> Tuple[bool, float]:
        """
        Monitor a position minute by minute throughout the hour
        
        :param symbol: Trading pair symbol
        :param entry_price: Position entry price
        :param position_type: 'long' or 'short'
        :param stop_loss_pct: Stop loss percentage as decimal
        :param current_time: Start time of the hour being tested
        :return: Tuple of (stop_loss_triggered: bool, exit_price: float)
        """
        try:
            # Fetch minute-by-minute data for the hour
            minute_data = self.fetch_minute_data(
                symbol=symbol,
                current_time=current_time,
                lookback_minutes=60
            )
            
            if not minute_data:
                # Fallback to hourly data if minute data isn't available
                current_price = self.get_price_at_time(symbol, current_time)
                if position_type == 'long':
                    price_decrease = (entry_price - current_price) / entry_price
                    stop_loss_triggered = price_decrease > stop_loss_pct
                else:  # short position
                    price_increase = (current_price - entry_price) / entry_price
                    stop_loss_triggered = price_increase > stop_loss_pct
                return stop_loss_triggered, current_price
                
            # Monitor each minute's data
            for ohlcv in minute_data:
                timestamp, open_price, high, low, close, volume = ohlcv
                print(timestamp, open_price, high, low, close, volume)
                # Check both high and low prices for stop loss violations
                prices_to_check = [high, low] if position_type == 'short' else [low, high]
                
                for price in prices_to_check:
                    if position_type == 'long':
                        price_decrease = (entry_price - price) / entry_price
                        if price_decrease > stop_loss_pct:
                            return True, price
                    else:  # short position
                        price_increase = (price - entry_price) / entry_price
                        if price_increase > stop_loss_pct:
                            return True, price
            
            # If no stop loss was triggered, return the last close price
            return False, minute_data[-1][4]  # Last candle's close price
            
        except Exception as e:
            logging.error(f"Error monitoring position for {symbol} at {current_time}: {e}")
            return False, 0.0
    
    def get_real_time_price_data(self, symbol: str, current_time: datetime) -> Dict:
        """
        Get price data for a specific point in time
        
        :param symbol: Trading pair symbol
        :param current_time: Simulated current time being tested
        :return: Dictionary with price and market data
        """
        try:
            # Fetch data for the specific timestamp
            ticker = self.exchange.fetch_ticker(symbol)
            orderbook = self.exchange.fetch_order_book(symbol)
            
            return {
                'last_price': float(ticker['last']),
                'bid': float(ticker['bid']),
                'ask': float(ticker['ask']),
                'volume': float(ticker['quoteVolume']),
                'best_bid_size': float(orderbook['bids'][0][1]),
                'best_ask_size': float(orderbook['asks'][0][1]),
                'timestamp': current_time.isoformat()
            }
        except Exception as e:
            print(e)
            return {}

    def get_price_at_time(self, symbol: str, timestamp: datetime) -> float:
        """
        Get the price at a specific timestamp from historical data
        
        :param symbol: Trading pair symbol
        :param timestamp: Specific timestamp to get price for
        :return: Price at that timestamp
        """
        try:
            # This method should be implemented to fetch from your historical price data
            # For your simulator, this would likely pull from your CSV file
            # Example implementation:
            df = pd.read_csv("Bitstamp_BTCUSD_1h_5.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            price = df[df['timestamp'] == timestamp]['close'].iloc[0]
            return float(price)
        except Exception as e:
            logging.error(f"Error getting price for {symbol} at {timestamp}: {e}")
            return 0.0

# In-memory cache for exchange rates
exchange_rate_cache = {}
CACHE_TTL = 60  # Time-to-live for cache entries in seconds

def get_total_volume(exchange):
    """
    Calculate the total 24h trading volume for an exchange.
    
    :param exchange: Instance of a CCXT exchange
    :return: Total trading volume as float
    """
    VOLUME_THRESHOLD = 2_000_000  # Adjust this value as needed
    try:
        markets = exchange.load_markets()
        total_volume = 0.0
        for symbol in markets:
            ticker = exchange.fetch_ticker(symbol)
            # Use quote volume as a proxy; adjust if needed
            volume = ticker.get('quoteVolume', 0.0)
            if volume:
                total_volume += float(volume)
                if total_volume >= VOLUME_THRESHOLD:
                    logging.info(f"Exchange '{exchange.id}' volume {total_volume} exceeds threshold {VOLUME_THRESHOLD}. Excluding from low liquidity selection.")
                    return float('inf')  # Assign a high volume to exclude this exchange
        return total_volume
    except Exception as e:
        logging.warning(f"Could not fetch volume for {exchange.id}: {e}")
        return float('inf')  # Assign a high volume to exclude problematic exchanges


def select_low_liquidity_exchanges(desired_count=10):
    """
    Select exchanges with the lowest liquidity based on 24h trading volume.
    
    :param desired_count: Number of low liquidity exchanges to select
    :return: List of exchange IDs
    """
    available_exchanges = ccxt.exchanges
    exchange_volumes = {}

    for idx, exchange_id in enumerate(available_exchanges):
        logging.info(f"Processing exchange {idx + 1}/{len(available_exchanges)}: {exchange_id}")
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,  # Respect rate limits
            })
            exchange.load_markets()
            total_volume = get_total_volume(exchange)
            exchange_volumes[exchange_id] = total_volume
            logging.info(f"Exchange: {exchange_id}, Total Volume: {total_volume}")
        except Exception as e:
            logging.warning(f"Failed to process {exchange_id}: {e}")
            exchange_volumes[exchange_id] = float('inf')  # Exclude problematic exchanges

        # Be polite and wait to respect rate limits
        time.sleep(exchange.rateLimit / 1000)

    # Sort exchanges by total_volume in ascending order and select the top 'desired_count'
    sorted_exchanges = sorted(exchange_volumes.items(), key=lambda item: item[1])
    low_liquidity_exchanges = [exchange_id for exchange_id, volume in sorted_exchanges[:desired_count]]

    logging.info(f"Selected Low Liquidity Exchanges: {low_liquidity_exchanges}")
    return low_liquidity_exchanges

    
def get_usd_to_quote_rate(exchange: ccxt.Exchange, quote_currency: str) -> Decimal:
    """
    Fetch the USD to Quote Currency conversion rate.
    
    :param exchange: CCXT exchange instance
    :param quote_currency: The quote currency (e.g., 'EUR')
    :return: Conversion rate as Decimal
    """
    cache_key = f"USD/{quote_currency}"
    current_time = time.time()
    
    # Check if the rate is cached and still valid
    if cache_key in exchange_rate_cache:
        rate, timestamp = exchange_rate_cache[cache_key]
        if current_time - timestamp < CACHE_TTL:
            logging.info(f"Using cached exchange rate for {cache_key}: {rate}")
            return rate
    
    # Attempt to fetch the conversion rate
    try:
        # Some exchanges might use 'EUR/USD' instead of 'USD/EUR'
        try:
            ticker = exchange.fetch_ticker(f"USD/{quote_currency}")
            rate = Decimal(str(ticker['last']))
            logging.info(f"Fetched rate {rate} for {cache_key} from USD/{quote_currency}")
        except ccxt.BadSymbol:
            # Try the inverse pair
            ticker = exchange.fetch_ticker(f"{quote_currency}/USD")
            inverse_rate = Decimal(str(ticker['last']))
            rate = Decimal('1') / inverse_rate
            logging.info(f"Fetched rate {rate} for {cache_key} from {quote_currency}/USD (inverse)")
        
        # Cache the fetched rate
        exchange_rate_cache[cache_key] = (rate, current_time)
        return rate
    except Exception as e:
        logging.error(f"Error fetching USD to {quote_currency} rate: {e}")
        return Decimal('0.0')  # Return zero if unable to fetch rate

class PriceMonitorRunner:
    def __init__(self, exchange: ccxt.Exchange):
        self.price_monitor = PriceMonitor(exchange)

    def run_monitor(self, symbol: str, entry_price: float, position_type: str,
                    stop_loss_pct: float, start_time: datetime) -> Tuple[bool, float]:
        """
        Monitor a position in real-time for one hour, checking every minute

        :param symbol: Trading pair symbol
        :param entry_price: Position entry price
        :param position_type: 'long' or 'short'
        :param stop_loss_pct: Stop loss percentage as decimal
        :param start_time: Start time of monitoring
        :return: Tuple of (stop_loss_triggered: bool, exit_price: float)
        """
        retries = 5

        def execute_monitoring(symbol: str, entry_price: float, position_type: str,
                    stop_loss_pct: float, start_time: datetime):
            """Inner function to encapsulate monitoring logic with retry support."""
            if isinstance(start_time, str):
                # Truncate any extra precision after microseconds
                if '.' in start_time:
                    base, fraction = start_time.split('.')
                    fraction = fraction[:6]  # Keep only up to 6 digits for microseconds
                    start_time = f"{base}.{fraction}"
                start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
            
            print(f"Parsed start_time: {start_time}")    

            end_time = start_time + timedelta(minutes=58)
            current_time = start_time

            while current_time < end_time:                    
                price_data = self.price_monitor.get_real_time_price_data(symbol, current_time)
                print(price_data)
                print(current_time)
                current_price = price_data['last_price']

                if position_type == 'long':
                    price_decrease = (float(entry_price) - float(current_price)) / float(entry_price)
                    if abs(price_decrease) > stop_loss_pct:
                        return True, current_price
                else:  # short position
                    price_increase = (float(current_price) - float(entry_price)) / float(entry_price)
                    if abs(price_increase) > stop_loss_pct:
                        return True, current_price

                time.sleep(60)
                current_time = datetime.now()

            # If we've completed the hour without triggering stop loss,
            # return the last known price
            final_data = self.price_monitor.get_real_time_price_data(symbol, current_time)
            return False, final_data.get('last_price', 0.0)

        for attempt in range(retries):
            try:
                return execute_monitoring(symbol, entry_price, position_type, stop_loss_pct, start_time)
            except Exception as e:
                print(f"Exception occurred on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    print("Retrying...")
                    time.sleep(5)  # Short delay before retrying
                else:
                    print("Max retries reached. Exiting monitoring.")
                    return False, 0.0
