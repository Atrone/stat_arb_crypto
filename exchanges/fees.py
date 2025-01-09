import ccxt
import logging
from typing import Tuple

def get_exchange_fees(exchange: ccxt.Exchange) -> Tuple[float, float]:
    """
    Retrieve the trading fees for a given exchange.

    :param exchange: CCXT exchange instance
    :return: Tuple of (maker_fee, taker_fee)
    """
    try:
        fees = exchange.fetch_trading_fees()
        # Assuming 'info' contains 'maker' and 'taker' fees
        maker_fee = fees.get('maker', 0.0001)  # Default to 0.01% if not available
        taker_fee = fees.get('taker', 0.0001)  # Default to 0.01% if not available
        return maker_fee, taker_fee
    except Exception as e:
        logging.error(f"Error fetching fees for {exchange.id}: {e}")
        return 0.0001, 0.0001  # Default fees