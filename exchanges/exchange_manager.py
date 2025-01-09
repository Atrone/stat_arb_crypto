import ccxt
import logging
from itertools import combinations
from typing import Set, Dict, Tuple

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