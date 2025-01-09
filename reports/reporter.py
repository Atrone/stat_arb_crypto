import logging
from typing import Dict, Tuple

def generate_report(report: Dict[Tuple[str, Tuple[str, str]], Dict[str, float]]) -> None:
    """
    Generate and display the comprehensive arbitrage report.

    :param report: Dictionary containing arbitrage results
    """
    logging.info("\nComprehensive Arbitrage Report:")
    for (symbol, exchange_pair), result in report.items():
        logging.info(
            f"Pair: {symbol} | Exchanges: {exchange_pair} | "
            f"Total Profit: {result['profit']:.2f} | Total Trades: {result['trades']}"
        )

def generate_summary(report: Dict[Tuple[str, Tuple[str, str]], Dict[str, float]]) -> None:
    """
    Generate and display summary statistics of the arbitrage simulation.

    :param report: Dictionary containing arbitrage results
    """
    overall_profit = sum([result['profit'] for result in report.values()])
    overall_trades = sum([result['trades'] for result in report.values()])
    logging.info(f"\nOverall Total Profit: {overall_profit:.2f}")
    logging.info(f"Overall Total Trades: {overall_trades}")