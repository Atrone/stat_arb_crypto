import ccxt
import logging
from decimal import Decimal
from .data_fetcher import get_usd_to_quote_rate

def estimate_slippage(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_size_usd: float,
    side: str = 'buy',
) -> float:
    """
    Estimate the slippage for a given trade on an exchange using a simplified approach.
    
    :param exchange: CCXT exchange instance
    :param symbol: Trading symbol (e.g., 'ETH/EUR' or 'BTC/USD')
    :param trade_size_usd: Trade size in USD
    :param side: 'buy' or 'sell'
    :return: Slippage as a decimal (e.g., 0.001 for 0.1%)
    """
    try:
        # Fetch limited order book depth (e.g., 20 levels should be sufficient)
        order_book = exchange.fetch_order_book(symbol, limit=20)
        
        # Get market price and orders based on side
        orders = order_book['asks'] if side == 'buy' else order_book['bids']
        if not orders:
            return 0.0
            
        market_price = orders[0][0]  # Best available price
        
        # Calculate the quote currency amount needed
        if "USD" in symbol.split("/")[1]:
            trade_size_quote = trade_size_usd
        else:
            ticker = exchange.fetch_ticker(symbol)
            # Convert USD to quote currency using last price
            trade_size_quote = trade_size_usd / ticker['last']
        
        # Calculate weighted average price
        total_quantity = 0
        total_cost = 0
        
        for price, quantity in orders:
            order_cost = price * quantity
            if total_cost + order_cost >= trade_size_quote:
                # Calculate the remaining quantity needed
                remaining = trade_size_quote - total_cost
                if remaining > 0:
                    total_cost += price * (remaining / price)
                    total_quantity += remaining / price
                break
            total_cost += order_cost
            total_quantity += quantity
            
        if total_quantity == 0:
            return 0.0
            
        # Calculate average execution price
        avg_price = total_cost / total_quantity
        
        # Calculate slippage
        slippage = (avg_price - market_price) / market_price if side == 'buy' else (market_price - avg_price) / market_price
        
        return float(slippage)
        
    except Exception as e:
        logging.error(f"Error calculating slippage for {symbol}: {e}")
        return 0.0
