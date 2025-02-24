import ccxt
import logging
import time
import os
import json
from itertools import combinations
from typing import Set, Dict, Tuple

def get_past_month_hourly_data(exchange: ccxt.Exchange,
                               symbol: str = 'BTC/USDC:USDC',
                               cutoff: int = 1739383546,
                               limit: int = 100) -> list:
    """
    Fetch all hourly OHLCV candlestick data for a given symbol from the exchange,
    going back in time until a candle's timestamp is less than or equal to the cutoff.

    The rate limit is respected by sleeping between requests based on the calculated weight.
    (weight = 2 + limit/10, so with limit=100, sleep = 0.3 * (2 + 100/10) = 3.6 sec per request)

    :param exchange: CCXT exchange instance
    :param symbol: Trading symbol to fetch data for, default is 'BTC/USDC:USDC'
    :param cutoff: Unix timestamp (in seconds) to fetch data up to (older than or equal)
    :param limit: Number of candles per request (affects rate limit); default is 100.
    :return: List of OHLCV data (each element typically like [timestamp, open, high, low, close, volume])
    """
    all_ohlcv = []
    current_until = 1686772053
    
    while True:
        params = {}
        # If we already have a lower bound from a previous fetch, use it.
        if current_until is not None:
            params["until"] = current_until

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit, params=params)
        except Exception as e:
            logging.error(f"Error fetching OHLCV data for {symbol} on {exchange}: {e}")
            break

        if not ohlcv:
            # No more data returned.
            break

        all_ohlcv.extend(ohlcv)
        print(ohlcv)

        # Assume the API returns candles in ascending order (oldest first).
        latest_timestamp = ohlcv[-1][0]
        print(latest_timestamp)
        if latest_timestamp >= cutoff * 1000:
            # We have reached (or passed) the cutoff timestamp.
            break

        # Prepare for the next call: use one second before the earliest timestamp so far.
        current_until = ((latest_timestamp) / 1000) + 3600 * 100

        # Enforce rate limit based on weight.
        weight = 2 + (limit / 10)
        sleep_time = 0.3 * weight
        time.sleep(sleep_time)
    
    return all_ohlcv

def get_supported_symbols(exchange: ccxt.Exchange) -> Set[str]:
    """
    Fetch supported trading symbols for a given exchange.

    :param exchange: CCXT exchange instance
    :return: Set of supported symbols
    """
    try:
        return set(exchange.load_markets().keys())
    except Exception as e:
        logging.error(f"Error fetching markets for {exchange.name}: {e}")
        return set()

def identify_matching_pairs(exchange_symbols: Dict[str, Set[str]]) -> Dict[str, list]:
    """
    Identify matching trading pairs across exchanges.

    :param exchange_symbols: Dictionary with exchange names as keys and sets of symbols as values
    :return: Dictionary with symbols as keys and list of exchange pairs as values
    """
    matching_pairs = {}
    for (exchange1, symbols1), (exchange2, symbols2) in combinations(exchange_symbols.items(), 2):
        common_symbols = symbols1 & symbols2  # Intersection of symbols
        for symbol in common_symbols:
            if symbol not in matching_pairs:
                matching_pairs[symbol] = []
            matching_pairs[symbol].append((exchange1, exchange2))
    return matching_pairs

if __name__ == "__main__":
    # Create an instance of the exchange.
    exchange = ccxt.vertex()
    symbol = 'BTC/USDC:USDC'
    cutoff = 1740326752  # The unix timestamp up to which we want to gather candlestick data.
    
    # Fetch candlestick data from the API.
    new_candles = get_past_month_hourly_data(exchange, symbol, cutoff, limit=100)
    
    # Define the file path to store the data.
    file_path = "ohlcv_data.txt"
    
    # Load existing candlestick data if the file already exists.
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    
    # Append the new candlesticks to the existing data.
    all_data = existing_data + new_candles
    
    # Write the combined data back to the file.
    with open(file_path, "w") as f:
        json.dump(all_data, f)
    
    total = len(all_data)
    print(f"Fetched {len(new_candles)} new candlesticks. Total stored: {total}.")