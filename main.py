import logging

from config import (
    ARBITRAGE_THRESHOLD,
    IS_STAT,
    STAT_SYMBOL,
    TRADE_SIZE,
    EXCHANGE_INSTANCES
)
from exchanges.exchange_manager import get_supported_symbols, identify_matching_pairs
from reports.reporter import generate_report, generate_summary

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def stat_main():
    exchange_symbols = {name: get_supported_symbols(instance) for name, instance in EXCHANGE_INSTANCES.items()}
    for exchange, symbols in exchange_symbols.items():
        for symbol in symbols:
            print(symbol)
            logging.info(f"Exchange: {exchange}, Symbol: {symbol}")
            if symbol == STAT_SYMBOL:
                logging.info(f"Exchange: {exchange}, Symbol: {symbol}")
                break
            # Process each symbol
    pass


def main():
    if IS_STAT:
        stat_main()
        return
    
    # TODO PURE ARBITRAGE

    # Fetch supported symbols for each exchange
    exchange_symbols = {name: get_supported_symbols(instance) for name, instance in EXCHANGE_INSTANCES.items()}

    
    matching_pairs = identify_matching_pairs(exchange_symbols)

    # Comprehensive report storage
    report = {}

    # Run the arbitrage simulation for all matching pairs
    for symbol, exchange_pairs in matching_pairs.items():
        for exchange_pair in exchange_pairs:
            logging.info(f"Simulating arbitrage for {symbol} on {exchange_pair}")

    # Display comprehensive report
    generate_report(report)

    # Summary statistics
    generate_summary(report)

if __name__ == "__main__":
    import sys
    print(sys.executable)
    main()