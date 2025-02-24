import csv
import json
import concurrent.futures
import statistics
import time
from datetime import timedelta, datetime
from random import random

import numpy as np
import pandas as pd
from ESRNN import ESRNN
from ESRNN.utils_evaluation import smape
import requests
import subprocess
from typing import Tuple, Dict
import sys
import os

PYTHON_PATH = r"C:\Users\antho\PycharmProjects\stat-arb-crypto\.venv\Scripts\python.exe"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = PROJECT_ROOT.replace("\\", "/")

def initialize_exchange(exchange_name: str) -> Dict:
    """
    Initialize a CCXT exchange instance using Python 3.8
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'''
import ccxt
import json
exchange = getattr(ccxt, "{exchange_name}")()
# Convert exchange instance to serializable dict
result = {{
    "id": exchange.id,
    "name": exchange.name if hasattr(exchange, "name") else exchange.id,
    "urls": exchange.urls,
    "api": exchange.api if hasattr(exchange, "api") else {{}},
    "has": exchange.has,
    "timeframes": exchange.timeframes if hasattr(exchange, "timeframes") else {{}},
}}
print(json.dumps(result))
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout.strip())
    except Exception as e:
        print(f"Error initializing exchange {exchange_name}: {e}")
        return {}

# Initialize exchanges using Python 3.8
EXCHANGE_INSTANCES = {
    exchange: initialize_exchange(exchange)
    for exchange in ['vertex', 'bitstamp']  # Use the same exchanges from config
}

def call_external_function(function_name: str, *args) -> Tuple[float, float]:
    """
    Call external Python 3.8 functions using subprocess from specific venv
    Returns tuple of (result, 0.0) or (0.0, 0.0) on error
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'from exchanges.data_fetcher import PriceMonitor\n'
            f'from exchanges.slippage import estimate_slippage\n'
            f'from exchanges.fees import get_exchange_fees\n'
            f'import ccxt\n'
            f'exchange = getattr(ccxt, "{args[0]["id"]}")() if "{args[0]["id"]}" in args[0] else args[0]\n'
            f'result = {function_name}(exchange, *args[1:])\n'
            f'print(result)'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip()), 0.0
    except Exception as e:
        print(f"Error calling {function_name}: {e}")
        return 0.0, 0.0

def create_price_monitor(exchange_instance: Dict) -> object:
    """
    Create a PriceMonitor instance using Python 3.8
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'''
import sys
sys.path.append(r"{PROJECT_ROOT}")
import json
from exchanges.data_fetcher import PriceMonitor
import ccxt

try:
    exchange = getattr(ccxt, "{exchange_instance['id']}")()
    monitor = PriceMonitor(exchange)
    print("MONITOR_CREATED")
except Exception as e:
    import traceback
    print("ERROR:", str(e))
    print(traceback.format_exc())
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            return False
        if "MONITOR_CREATED" in result.stdout:
            return True
        return False
    except Exception as e:
        print(f"Error creating price monitor: {e}")
        return False

def get_exchange_fees(exchange_instance: Dict) -> Tuple[float, float]:
    """
    Get exchange fees using Python 3.8
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'''
import sys
sys.path.append(r"{PROJECT_ROOT}")
from exchanges.fees import get_exchange_fees
import ccxt

try:
    exchange = getattr(ccxt, "{exchange_instance['id']}")()
    fees = get_exchange_fees(exchange)
    print(fees[0])
except Exception as e:
    import traceback
    print("ERROR:", str(e))
    print(traceback.format_exc())
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            return 0.0, 0.0
        try:
            return float(result.stdout.strip().split('\n')[-1]), 0.0
        except:
            print(f"Unexpected output format: {result.stdout}")
            return 0.0, 0.0
    except Exception as e:
        print(f"Error getting exchange fees: {e}")
        return 0.0, 0.0

def estimate_slippage(exchange_instance: Dict, symbol: str, price: float, side: str) -> float:
    """
    Estimate slippage using Python 3.8
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'''
import sys
sys.path.append(r"{PROJECT_ROOT}")
from exchanges.slippage import estimate_slippage
import ccxt

try:
    exchange = getattr(ccxt, "{exchange_instance['id']}")()
    slippage = estimate_slippage(exchange, "{symbol}", {price}, "{side}")
    print(slippage)
except Exception as e:
    import traceback
    print("ERROR:", str(e))
    print(traceback.format_exc())
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            return 0.0
        try:
            return float(result.stdout.strip().split('\n')[-1])
        except:
            print(f"Unexpected output format: {result.stdout}")
            return 0.0
    except Exception as e:
        print(f"Error estimating slippage: {e}")
        return 0.0

def monitor_position(exchange_instance: Dict, symbol: str, entry_price: float, 
                    position_type: str, stop_loss_pct: float, 
                    current_time: str) -> Tuple[bool, float]:
    """
    Monitor a position using Python 3.8
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'''
import sys
sys.path.append(r"{PROJECT_ROOT}")
import json
from exchanges.data_fetcher import PriceMonitor
import ccxt

try:
    exchange = getattr(ccxt, "{exchange_instance['id']}")()
    monitor = PriceMonitor(exchange)
    result = monitor.monitor_position(
        symbol="{symbol}",
        entry_price={entry_price},
        position_type="{position_type}",
        stop_loss_pct={stop_loss_pct},
        current_time="{current_time}"
    )
    print(json.dumps(result))
except Exception as e:
    import traceback
    print("ERROR:", str(e))
    print(traceback.format_exc())
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            return False, 0.0
        try:
            stop_triggered, exit_price = json.loads(result.stdout.strip().split('\n')[-1])
            return bool(stop_triggered), float(exit_price)
        except:
            print(f"Unexpected output format: {result.stdout}")
            return False, 0.0
    except Exception as e:
        print(f"Error monitoring position: {e}")
        return False, 0.0

def get_average_funding_rate(max_time):
    """
    Fetches the average funding rate for 60 minutes before the max_time inputted.
    """
    time.sleep(20)
    base_url = "https://archive.prod.vertexprotocol.com/v1"
    max_time_dt = datetime.fromisoformat(max_time.replace("Z", "+00:00"))
    max_time_unix = int(max_time_dt.timestamp())
    
    body = {
        "market_snapshots": {
            "interval": {
                "count": 3,
                "granularity": 3000,
                "max_time": max_time_unix
            },
            "product_ids": [2]  # BTC product ID
        }
    }

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    response = requests.post(base_url, json=body, headers=headers, verify=False)

    if response.status_code != 200:
        return 0.0  # Return neutral funding rate on error

    data = response.json()
    funding_rates = []
    if 'snapshots' in data:
        for snapshot in data['snapshots']:
            if 'funding_rates' in snapshot and '2' in snapshot['funding_rates']:
                funding_rates.append(float(snapshot['funding_rates']['2']))

    return sum(funding_rates) / len(funding_rates) if funding_rates else 0.0

def get_past_month_hourly_data_subprocess(exchange_instance: dict, symbol: str = 'BTC/USDC:USDC') -> list:
    """
    Retrieve the past month's hourly OHLCV data using the exchange manager's function,
    via a subprocess call (using the same technique as other functions in this file).

    :param exchange_instance: Dictionary representing the exchange instance (must include "id")
    :param symbol: Trading symbol (default 'BTC/USDC:USDC')
    :return: List of OHLCV data, or an empty list on error.
    """
    import json
    import subprocess

    try:
        cmd = [
            PYTHON_PATH,
            "-c",
            f'''
import sys
sys.path.append(r"{PROJECT_ROOT}")
import ccxt, json
from exchanges.exchange_manager import get_past_month_hourly_data
exchange = getattr(ccxt, "{exchange_instance['id']}")()
data = get_past_month_hourly_data(exchange, "{symbol}")
print(json.dumps(data))
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout.strip())
        return data
    except Exception as e:
        print(f"Error retrieving hourly data via subprocess: {e}")
        return []

def simulate(exchange_name: str, symbol: str):
    """
    Run the statistical arbitrage simulation with stop loss logic commented out.
    """
    print(f"Starting simulation for exchange: {exchange_name} and symbol: {symbol}")
    fear_greed = requests.get('https://api.alternative.me/fng/?limit=0&format=csv').text.split("\n", 3)[3].split('\n')[:-2]
    a = []

    for fg in fear_greed[1:-3]:
        a.append({"date": datetime.strptime(fg.split(",")[0], "%d-%m-%Y"), "x": int(fg.split(",")[1])})

    # ----- Data Acquisition & Preparation -----
    # Instead of reading from a CSV, fetch Vertex data:
    with open(os.path.join(PROJECT_ROOT, "exchanges", "ohlcv_data.txt"), "r") as f:
        vertex_data = json.load(f)
    unique_vertex_data = {tuple(item) for item in vertex_data}
    vertex_data = sorted([list(item) for item in unique_vertex_data], key=lambda x: x[0])
    print(vertex_data)
    df = pd.DataFrame(vertex_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['ds'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df.ds.dt.date
    dfx = pd.DataFrame(a)
    dfx['date'] = dfx.date.astype('datetime64[ns]')
    df['date'] = df.date.astype('datetime64[ns]')
    df = df.merge(right=dfx, on='date', how='left')
    df['y'] = df['close']

    # ----- Data Splitting for ESRNN -----
    end = 10
    state = 0
    old_state = 0
    trades = 0
    stop_trigger_count = 0
    failures = 0
    money = 0
    predictions = 0
    successes = 0
    failures_list = []
    successes_list = []
    smapes = []

    optimization_phase = 0  # focus on 0, then 500, then 1000, etc.

    # sim 1
    #6484.313917554924
    #425
    #92
    #88


    # sim 2
    #1917.8585406474976
    #423
    #53
    #49


    # sim 3
    #-4094.1455496996914
    #434
    #36
    #30

    # sim 4
    #3454.387306896452
    #422
    #27
    #28
    daily_returns = []
    daily_pnl = 0
    last_date = None


    for n in range(-170, 0 + optimization_phase, 1):
        x_train = df.drop(labels='y', axis=1)
        y_train = df.drop(labels='x', axis=1)
        x_train = x_train.iloc[:-(end - n)]
        y_train = y_train.iloc[:-(end - n)]
        x_train = x_train.reset_index().drop(labels='index', axis=1)
        y_train = y_train.reset_index().drop(labels='index', axis=1)
        x_test = df.drop(labels='y', axis=1)
        y_test = df.drop(labels='x', axis=1)
        x_test = x_test.iloc[:-(end * 5 - n)]
        y_test = y_test.iloc[:-(end * 5 - n)]
        x_test = x_test.reset_index().drop(labels='index', axis=1)
        y_test = y_test.reset_index().drop(labels='index', axis=1)
        y_test['y_hat_naive2'] = y_test['y'] + y_test['y'] * (random() / 100)
        y_train['unique_id'] = "BTC"
        x_train['unique_id'] = "BTC"
        y_test['unique_id'] = "BTC"
        x_test['unique_id'] = "BTC"
        print(x_train)
        print(y_train)
        print(x_test)
        print(y_test)

        # START TUNABLE PARAMETERS
        stop_loss_percentage = 0.0025
        stop_loss_percentage_long = 0.0025
        limit_sd = 1
        limit = statistics.mean(abs(y_train[:-1]['y'].pct_change().iloc[1:-1].values)) + statistics.stdev(
            abs(y_train[:-1]['y'].pct_change().iloc[1:-1].values)) * limit_sd
        funded_rate_sensitity_long = 11616242444237 + 5250911279712 * 1
        funded_rate_sensitity_short = -600077852205 - 1620712129010 * 1
        funded_rate_sensitity = 10128288448857.494141
        
        class constants:
            optimized_hyper_params_for_1_to_45 = {
                "max_epochs": 200, "learning_rate": 4.5e-3, "seasonality": [24], "lr_decay": 0.925,
                "lr_scheduler_step_size": 10,
                "per_series_lr_multip": 1.5,
                "freq_of_test": 5, "batch_size": 1, "output_size": 5
            }
            optimized_hyper_params_for_45_to_x = {
                "max_epochs": 300, "learning_rate": 4e-3, "seasonality": [24], "lr_decay": 0.95,
                "lr_scheduler_step_size": 20,
                "per_series_lr_multip": 1.5,
                "freq_of_test": 5, "batch_size": 1, "output_size": 5
            }
            optimized_hyper_params_for_comb_tried_but_didnt_work = {
                "max_epochs": 650, "learning_rate": 4e-3, "seasonality": [24], "lr_decay": 0.95,
                "lr_scheduler_step_size": 15,
                "per_series_lr_multip": 0.75,
                "freq_of_test": 5, "batch_size": 1, "output_size": 5
            }
            optimized_hyper_params_for_comb_the_one_i_just_tried = {
                "max_epochs": 500, "learning_rate": 4e-3, "seasonality": [24], "lr_decay": 0.95,
                "lr_scheduler_step_size": 15,
                "per_series_lr_multip": 0.5,
                "freq_of_test": 5, "batch_size": 1, "output_size": 5
            }
            optimized_hyper_params_for_comb = {
                "max_epochs": 275, "learning_rate": 4.5e-3, "seasonality": [24], "lr_decay": 0.95,
                "lr_scheduler_step_size": 20,
                "per_series_lr_multip": 0.5,
                "freq_of_test": 5, "batch_size": 1, "output_size": 5
            }
        # END TUNABLE PARAMETERS

        # ----- ESRNN Model Setup & Training -----
        model = ESRNN(**constants.optimized_hyper_params_for_comb_the_one_i_just_tried)
        print(x_train[:-1])
        print(y_train[:-1])
        print(x_test[:-1])
        print(y_test[:-1])
        model.fit(x_train[:-1], y_train[:-1], x_test[:-1], y_test[:-1], verbose=False)

        # ----- Forecasting & Simulation Loop -----
        listOfDates = []
        date = x_train[:-1]['ds'].iloc[-1]
        for i in range(5):
            date += timedelta(hours=1)
            listOfDates.append(date)
        data = [{"ds": i,
                 "x": int(df.iloc[n]['x']),
                 "unique_id": "BTC"} for i in listOfDates]
        y_hat_df = model.predict(pd.DataFrame(data))
        y_hat_df = y_hat_df.dropna()
        y_hat_df = y_hat_df.reset_index().drop(labels='index', axis=1)

        print(y_hat_df)
        ai_price = y_hat_df['y_hat'][0]
        print(df['ds'].iloc[-(end - n + 2)])
        print(df['ds'].iloc[-(end - n + 1)])
        real_price_before = df['close'].iloc[-(end - n + 2)]
        real_price_after = df['close'].iloc[-(end - n + 1)]
        print(real_price_before)
        print(real_price_after)
        print(limit)

        monitor_created = create_price_monitor(EXCHANGE_INSTANCES['vertex'])
        fees = get_exchange_fees(EXCHANGE_INSTANCES['vertex'])[0]
        slippage = estimate_slippage(EXCHANGE_INSTANCES['vertex'], symbol, real_price_before, 'buy')
        print(slippage)
        print(fees)
        current_time = str(df['ds'].iloc[-(end - n + 2)])
        avg_funding_rate = get_average_funding_rate(current_time.replace(" ", "T") + "Z")
        print(avg_funding_rate)
        print(current_time)

        if (abs(real_price_before - ai_price) / real_price_before) > limit:
            continue

        if real_price_before < ai_price and avg_funding_rate > funded_rate_sensitity_long: # if everyone says long, and we say long, go short
            state = 1
            fees = 0
            slippage = 0
            if state != old_state:
                trades += 1
                fees = get_exchange_fees(EXCHANGE_INSTANCES['vertex'])[0]
                slippage = estimate_slippage(EXCHANGE_INSTANCES['vertex'], symbol, real_price_before, 'buy')
                old_state = state
            if real_price_before < real_price_after:
                money -= abs(real_price_after - real_price_before) - (real_price_before * fees) - (real_price_before * slippage)
                successes += 1
                failures_list.append(0)
                successes_list.append(1)
            else:
                # ----- Stop Loss Logic Commented Out -----
                # stop_triggered, exit_price = monitor_position(
                #     EXCHANGE_INSTANCES['vertex'],
                #     symbol="BTC/USDC:USDC",
                #     entry_price=real_price_before,
                #     position_type='short' if real_price_before > ai_price else 'long',
                #     stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                #     current_time=current_time
                # )
                # print(stop_triggered)
                # print(exit_price)
                # if stop_triggered:
                #     state = 0
                #     fees = 0
                #     slippage = 0
                #     if state != old_state:
                #         fees = get_exchange_fees(EXCHANGE_INSTANCES['vertex'])[0]
                #         slippage = estimate_slippage(EXCHANGE_INSTANCES['vertex'], symbol, real_price_before, 'buy')
                #         trades += 1
                #         old_state = state
                #     stop_trigger_count += 1
                #     failures += 1
                #     money -= abs(exit_price - real_price_before) + (real_price_before * fees) + (real_price_before * slippage)
                #     continue
                money += abs(real_price_after - real_price_before) + (real_price_before * fees) + (real_price_before * slippage)
                failures += 1
                failures_list.append(1)
                successes_list.append(0)
        elif real_price_before > ai_price and avg_funding_rate < funded_rate_sensitity_short: # if everyone says short, and we say short, go long
            state = -1
            fees = 0
            slippage = 0
            if state != old_state:
                fees = get_exchange_fees(EXCHANGE_INSTANCES['vertex'])[0]
                slippage = estimate_slippage(EXCHANGE_INSTANCES['vertex'], symbol, real_price_before, 'buy')
                trades += 1
                old_state = state
            if real_price_before > real_price_after:
                money -= abs(real_price_before - real_price_after) - real_price_before * fees - real_price_before * slippage
                successes += 1
                failures_list.append(0)
                successes_list.append(1)
            else:
                # ----- Stop Loss Logic Commented Out -----
                # stop_triggered, exit_price = monitor_position(
                #     EXCHANGE_INSTANCES['vertex'],
                #     symbol="BTC/USDC:USDC",
                #     entry_price=real_price_before,
                #     position_type='short' if real_price_before > ai_price else 'long',
                #     stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                #     current_time=current_time
                # )
                # print(stop_triggered)
                # print(exit_price)
                # if stop_triggered:
                #     state = 0
                #     fees = 0
                #     slippage = 0
                #     if state != old_state:
                #         trades += 1
                #         old_state = state
                #     stop_trigger_count += 1
                #     failures += 1
                #     money -= abs(exit_price - real_price_before) + (real_price_before * fees) + (real_price_before * slippage)
                #     continue
                money += abs(real_price_before - real_price_after) + (real_price_before * fees) + (real_price_before * slippage)
                failures += 1
                failures_list.append(1)
                successes_list.append(0)
        else:
            state = 0
            fees = 0
            slippage = 0
            if state != old_state:
                trades += 1
                fees = get_exchange_fees(EXCHANGE_INSTANCES['vertex'])[0]
                slippage = estimate_slippage(EXCHANGE_INSTANCES['vertex'], symbol, real_price_before, 'buy')
                old_state = state
            # sell everything here
            pass
        
        current_date = df['ds'].iloc[-(end - n + 2)].date()
        
        # Track PnL for daily returns calculation
        trade_pnl = 0
        if real_price_before < ai_price and avg_funding_rate > funded_rate_sensitity_long:
            if real_price_before < real_price_after:
                trade_pnl = -(abs(real_price_after - real_price_before) + (real_price_before * fees) + (real_price_before * slippage))
            else:
                trade_pnl = abs(real_price_after - real_price_before) - (real_price_before * fees) - (real_price_before * slippage)
        elif real_price_before > ai_price and avg_funding_rate < funded_rate_sensitity_short:
            if real_price_before > real_price_after:
                trade_pnl = -(abs(real_price_before - real_price_after) + (real_price_before * fees) + (real_price_before * slippage))
            else:
                trade_pnl = abs(real_price_before - real_price_after) - (real_price_before * fees) - (real_price_before * slippage)
        
        # Calculate daily returns
        if last_date is None:
            last_date = current_date
            daily_pnl = trade_pnl
        elif current_date != last_date:
            # Calculate return as percentage of initial price
            daily_return_pct = (daily_pnl / real_price_before) * 100
            daily_returns.append(daily_return_pct)
            # Reset for next day
            daily_pnl = trade_pnl
            last_date = current_date
        else:
            daily_pnl += trade_pnl

        predictions += 1
        print(stop_trigger_count)
        print(smape(ai_price, real_price_after))
        smapes.append(smape(ai_price, real_price_after))
        print(trades)
        print(money)
        print(predictions)
        print(successes)
        print(failures)
        print(successes_list)
        print(failures_list)
        print(smapes)
        print(daily_returns)
        
if __name__ == "__main__":
    simulate('vertex', 'BTC/USDC:USDC')
