from typing import Tuple
import ccxt  # Add this import

# Add exchange initialization
def get_cryptocom_exchange(api_key: str, api_secret: str) -> ccxt.cryptocom:
    """
    Initialize Crypto.com exchange with API credentials
    """
    return ccxt.cryptocom({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })

def execute_margin_trade(symbol: str, side: str, amount: float, price: float) -> Tuple[bool, str]:
    """
    Execute a margin trade on the exchange
    """
    exchange = get_cryptocom_exchange('', '')
    exchange.options['defaultType'] = 'margin'
    try:
        # Create the margin order
        order = exchange.create_order(
            symbol=symbol,
            type='market',  # or 'limit' depending on your needs
            side=side,
            amount=amount,
            price=price
        )
        
        return True, order['id']
    except Exception as e:
        print(f"Error in execute_margin_trade: {e}")
        return False, ""

# ---- Duplicated functions for Vertex exchange ----

def get_vertex_exchange(api_key: str, api_secret: str) -> ccxt.vertex:
    """
    Initialize Vertex exchange with API credentials
    """
    return ccxt.vertex({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })

def execute_margin_trade_vertex(symbol: str, side: str, amount: float, price: float) -> Tuple[bool, str]:
    """
    Execute a margin trade on the Vertex exchange
    """
    exchange = get_vertex_exchange('', '')
    exchange.options['defaultType'] = 'margin'
    try:
        # Create the margin order
        order = exchange.create_order(
            symbol=symbol,
            type='market',  # adjust this as necessary for your use case
            side=side,
            amount=amount,
            price=price
        )
        return True, order['id']
    except Exception as e:
        print(f"Error in execute_margin_trade_vertex: {e}")
        return False, ""