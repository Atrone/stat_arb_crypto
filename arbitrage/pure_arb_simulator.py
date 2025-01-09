import logging
from decimal import Decimal
from typing import Tuple
import pandas as pd
from tenacity import retry, wait_fixed, stop_after_attempt
import time
import ccxt
from exchanges.fees import get_exchange_fees
from exchanges.slippage import estimate_slippage
from exchanges.data_fetcher import get_usd_to_quote_rate
from config import IS_STAT, TIMEFRAME

def is_spot_market(exchange, symbol):
    try:
        markets = exchange.load_markets()
        if symbol in markets:
            return markets[symbol].get('spot', False)
        return False
    except Exception as e:
        logging.error(f"Error loading markets for {exchange.id}: {e}")
        return False

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def fetch_data(exchange: ccxt.Exchange, symbol: str, timeframe: str, since=None, limit=5) -> pd.Series:
    """
    Fetch historical closing price data for a given symbol from an exchange.

    :param exchange: CCXT exchange instance
    :param symbol: Trading symbol
    :param timeframe: Timeframe for OHLCV data
    :param since: Timestamp to fetch data from
    :param limit: Number of data points to fetch
    :return: Pandas Series of closing prices
    """
    if not is_spot_market(exchange, symbol) and not IS_STAT:
        logging.warning(f"Symbol {symbol} is not a spot market on {exchange.name}. Skipping OHLCV fetch.")
        return None  # Or handle accordingly

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        time.sleep(exchange.rateLimit / 1000)  # Respect rate limits    
        return df['close']
    except ccxt.BadRequest as e:
        logging.error(f"BadRequest for symbol {symbol} on exchange {exchange.name}: {e}")
        raise e  # Let tenacity handle the retry
    except Exception as e:
        logging.error(f"Error fetching data for symbol {symbol} on exchange {exchange.name}: {e}")
        raise e

def simulate_arbitrage(
    symbol: str,
    exchanges_pair: Tuple[str, str],
    trade_size_usd: float,
    exchange_instances: dict,
    arbitrage_threshold: float = 0
) -> Tuple[float, int]:
    """
    Simulate arbitrage between two exchanges in both directions based on the given symbol and trade size in USD.
    
    :param symbol: Trading symbol (e.g., 'ETH/EUR' or 'BTC/USD')
    :param exchanges_pair: Tuple containing (exchange1, exchange2)
    :param trade_size_usd: Trade size in USD
    :return: Tuple containing total profit and number of trades executed
    """
    buy_exchange_name, sell_exchange_name = exchanges_pair
    buy_exchange = exchange_instances[buy_exchange_name]
    sell_exchange = exchange_instances[sell_exchange_name]

    # Retrieve fees
    buy_maker_fee, buy_taker_fee = get_exchange_fees(buy_exchange)
    sell_maker_fee, sell_taker_fee = get_exchange_fees(sell_exchange)
    total_fees = buy_taker_fee + sell_taker_fee  # Assuming taker fees for immediate execution

    logging.info(f"Simulating arbitrage for {symbol} between {buy_exchange_name} and {sell_exchange_name}...")

    # Fetch and merge historical data
    data_buy = fetch_data(buy_exchange, symbol, TIMEFRAME)
    data_sell = fetch_data(sell_exchange, symbol, TIMEFRAME)
    if data_buy is None or data_sell is None:
        logging.warning(f"Skipping arbitrage simulation for {symbol} between {buy_exchange_name} and {sell_exchange_name} due to missing data.")
        return 0.0, 0
    combined_data = pd.concat([data_buy, data_sell], axis=1, keys=[buy_exchange_name, sell_exchange_name]).dropna()
    combined_data.index = pd.to_datetime(combined_data.index, utc=True)
    combined_data = combined_data.sort_index().dropna()
    profit = 0.0
    num_trades = 0

    for timestamp, prices in combined_data.iterrows():
        buy_price = prices[buy_exchange_name]
        sell_price = prices[sell_exchange_name]
        logging.info(f"Buy price on {buy_exchange_name}: {buy_price:.5f}")
        logging.info(f"Sell price on {sell_exchange_name}: {sell_price:.5f}")

        # Estimate slippage for buying and selling
        buy_slippage = estimate_slippage(buy_exchange, symbol, trade_size_usd, side='buy')
        sell_slippage = estimate_slippage(sell_exchange, symbol, trade_size_usd, side='sell')

        logging.info(f"Buy slippage: {buy_slippage * 100:.2f}%")
        logging.info(f"Sell slippage: {sell_slippage * 100:.2f}%")

        # Direction 1: Buy on Exchange A, Sell on Exchange B
        adjusted_buy_price_A = buy_price * (1 + buy_slippage)
        adjusted_sell_price_A = sell_price * (1 - sell_slippage)

        # Calculate potential profit before fees
        potential_profit_A = Decimal(str(adjusted_sell_price_A)) - Decimal(str(adjusted_buy_price_A))
        logging.info(f"Potential profit (Buy on {buy_exchange_name}, Sell on {sell_exchange_name}): {potential_profit_A:.10f}")

        # Calculate fee costs for Direction 1
        fee_cost_buy_A = Decimal(str(adjusted_buy_price_A)) * Decimal(str(buy_taker_fee))
        fee_cost_sell_A = Decimal(str(adjusted_sell_price_A)) * Decimal(str(sell_taker_fee))
        fee_cost_A = fee_cost_buy_A + fee_cost_sell_A
        logging.info(f"Fee cost Direction 1: {fee_cost_A:.10f}")

        # Calculate net profit for Direction 1
        net_profit_A = potential_profit_A - fee_cost_A

        # Initialize currency variables
        try:
            base_currency, quote_currency = symbol.split('/')
        except Exception as e:
            logging.error(f"Error parsing symbol '{symbol}': {e}")
            continue  # Skip to next iteration

        # Convert net profit to USD if necessary
        if quote_currency != 'USD':
            converter_exchange = ccxt.kraken()
            converter_exchange.load_markets()
            usd_to_quote_rate = get_usd_to_quote_rate(converter_exchange, quote_currency)

            if usd_to_quote_rate != Decimal('0.0'):
                net_profit_usd_A = net_profit_A * usd_to_quote_rate
            else:
                logging.error(f"USD to {quote_currency} rate is zero; cannot convert net profit to USD for Direction 1.")
                net_profit_usd_A = Decimal('0.0')
        else:
            net_profit_usd_A = net_profit_A

        if net_profit_usd_A > Decimal(str(arbitrage_threshold)):
            profit += float(net_profit_usd_A)
            num_trades += 1
            logging.info(
                f"{timestamp}: Direction 1 - Buy on {buy_exchange_name} at {adjusted_buy_price_A:.5f} {quote_currency} "
                f"(slippage: {buy_slippage * 100:.2f}%), Sell on {sell_exchange_name} at {adjusted_sell_price_A:.5f} {quote_currency} "
                f"(slippage: {sell_slippage * 100:.2f}%). Net Profit: {net_profit_usd_A:.5f} USD"
            )
        else:
            logging.info(f"Direction 1 - No profitable arbitrage opportunity. Net Profit: {net_profit_usd_A:.5f} USD")

        # Direction 2: Buy on Exchange B, Sell on Exchange A
        adjusted_buy_price_B = sell_price * (1 + sell_slippage)
        adjusted_sell_price_B = buy_price * (1 - buy_slippage)

        # Calculate potential profit before fees
        potential_profit_B = Decimal(str(adjusted_sell_price_B)) - Decimal(str(adjusted_buy_price_B))
        logging.info(f"Potential profit (Buy on {sell_exchange_name}, Sell on {buy_exchange_name}): {potential_profit_B:.10f}")

        # Calculate fee costs for Direction 2
        fee_cost_buy_B = Decimal(str(adjusted_buy_price_B)) * Decimal(str(sell_taker_fee))
        fee_cost_sell_B = Decimal(str(adjusted_sell_price_B)) * Decimal(str(buy_taker_fee))
        fee_cost_B = fee_cost_buy_B + fee_cost_sell_B
        logging.info(f"Fee cost Direction 2: {fee_cost_B:.10f}")

        # Calculate net profit for Direction 2
        net_profit_B = potential_profit_B - fee_cost_B

        # Convert net profit to USD if necessary
        if quote_currency != 'USD':
            if usd_to_quote_rate != Decimal('0.0'):
                net_profit_usd_B = net_profit_B * usd_to_quote_rate
            else:
                logging.error(f"USD to {quote_currency} rate is zero; cannot convert net profit to USD for Direction 2.")
                net_profit_usd_B = Decimal('0.0')
        else:
            net_profit_usd_B = net_profit_B

        if net_profit_usd_B > Decimal(str(arbitrage_threshold)):
            profit += float(net_profit_usd_B)
            num_trades += 1
            logging.info(
                f"{timestamp}: Direction 2 - Buy on {sell_exchange_name} at {adjusted_buy_price_B:.5f} {quote_currency} "
                f"(slippage: {sell_slippage * 100:.2f}%), Sell on {buy_exchange_name} at {adjusted_sell_price_B:.5f} {quote_currency} "
                f"(slippage: {buy_slippage * 100:.2f}%). Net Profit: {net_profit_usd_B:.5f} USD"
            )
        else:
            logging.info(f"Direction 2 - No profitable arbitrage opportunity. Net Profit: {net_profit_usd_B:.5f} USD")

    return profit, num_trades
