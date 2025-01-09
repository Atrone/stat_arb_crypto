import csv
import json
import concurrent.futures
import statistics
import time
from datetime import timedelta, datetime
from random import random

import numpy as np
import pandas
from ESRNN import ESRNN
from ESRNN.utils_evaluation import smape
import requests
import subprocess
from typing import Tuple, Dict
import sys
import os
import test_perms

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


def get_cryptocom_exchange() -> Dict:
    """
    Initialize a CCXT exchange instance using Python 3.8
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'''
import ccxt
print(json.dumps(ccxt.cryptocom({
        'apiKey': 'MzdwNm6GSanC9B1N4bcfXy',
        'secret': 'cxakp_BGhPnLXC2F2isseJc8pSGh',
        'enableRateLimit': True
    })))'''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout.strip())
    except Exception as e:
        print(f"Error initializing exchange cc: {e}")
        return {}


# Initialize exchanges using Python 3.8
EXCHANGE_INSTANCES = {
    exchange: initialize_exchange(exchange)
    for exchange in ['cryptocom', 'bitstamp']  # Use the same exchanges from config
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
                    start_time: str) -> Tuple[bool, float]:
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
from exchanges.data_fetcher import PriceMonitorRunner
import ccxt

try:
    exchange = getattr(ccxt, "{exchange_instance['id']}")()
    monitor = PriceMonitorRunner(exchange)
    print("MONITOR_CREATED")
    result = monitor.run_monitor(
        symbol="{symbol}",
        entry_price={entry_price},
        position_type="{position_type}",
        stop_loss_pct={stop_loss_pct},
        start_time="{start_time}"
    )
    print(json.dumps(result))
except Exception as e:
    import traceback
    print("ERROR:", str(e))
    print(traceback.format_exc())
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Print all stdout lines
        if result.stdout:
            print("Subprocess output:")
            print(result.stdout)
            
        # Print all stderr lines
        if result.stderr:
            print("Subprocess errors:")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            return False, 0.0
            
        try:
            # Look for the line starting with "RESULT:" to parse the actual return value
            for line in result.stdout.strip().split('\n'):
                if line.startswith("RESULT:"):
                    stop_triggered, exit_price = json.loads(line[7:])  # Skip "RESULT:" prefix
                    return bool(stop_triggered), float(exit_price)
            print("No result found in output")
            return False, 0.0
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


def make_trade(symbol: str, side: str, amount: float, price: float) -> Tuple[bool, str]:
    """
    Execute a margin trade using Python 3.8
    Args:
        exchange_instance: Dictionary containing exchange details
        symbol: Trading pair symbol (e.g. 'BTC/USD')
        side: 'buy' or 'sell'
        amount: Amount to trade
        price: Price to trade at
    Returns:
        Tuple of (success: bool, order_id: str)
    """
    try:
        cmd = [
            PYTHON_PATH,
            '-c',
            f'''
import sys
sys.path.append(r"{PROJECT_ROOT}")
import json
from exchanges.trading import execute_margin_trade
import ccxt

try:    
    success, order_id = execute_margin_trade(
        symbol="{symbol}",
        side="{side}",
        amount={amount},
        price={price}
    )
    print(json.dumps([success, order_id]))
except Exception as e:
    import traceback
    print("ERROR:", str(e))
    print(traceback.format_exc())
            '''
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Print all stdout lines
        if result.stdout:
            print("Subprocess output:")
            print(result.stdout)
            
        # Print all stderr lines
        if result.stderr:
            print("Subprocess errors:")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            return False, 0.0
            
        try:
            # Look for the line starting with "RESULT:" to parse the actual return value
            for line in result.stdout.strip().split('\n'):
                if line.startswith("RESULT:"):
                    success, order_id = json.loads(line[7:])  # Skip "RESULT:" prefix
                    return bool(success), str(order_id)
            print("No result found in output")
            return False, 0.0
        except:
            print(f"Unexpected output format: {result.stdout}")
            return False, 0.0
    except Exception as e:
        print(f"Error executing trade: {e}")
        return False, ""

def state_to_action(state: int):
    if state == 1:
        test_perms.close_btcusd_positions()
        test_perms.open_btcusd_position("BUY", 0.05)
    elif state == -1:
        test_perms.close_btcusd_positions()
        test_perms.open_btcusd_position("SELL", 0.05)
    elif state == 0:
        test_perms.close_btcusd_positions()

def run(exchange, symbol):

    stop_loss_percentage = 0.005
    stop_loss_percentage_long = 0.01
    predictions = 0
    trades = 0
    old_state = 1
    state = 1
    stop_trigger_count = 0

    while True:
        fear_greed = requests.get('https://api.alternative.me/fng/?limit=0&format=csv').text.split("\n", 3)[3].split('\n')[:-2]
        a = []

        for fg in fear_greed[1:-3]:
            a.append({"date": datetime.strptime(fg.split(",")[0], "%d-%m-%Y"), "x": int(fg.split(",")[1])})

        import pandas as pd
        from tradingfeatures import bitstamp

        bitstamp = bitstamp()

        bitstamp.update('bitstamp.csv')

        # Assuming your CSV has a column with Unix timestamps
        # Read the CSV
        df = pd.read_csv('bitstamp.csv')


        # Convert Unix timestamp to datetime format
        # If your timestamp is in seconds
        df['datetime_column'] = pd.to_datetime(df['timestamp'], unit='s')

        # Format the datetime to the desired string format
        df['datetime_column'] = df['datetime_column'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Remove first 45k rows using iloc
        df = df[44924:]

        # Save without headers
        df.to_csv('output_file.csv', index=False, header=False)


        df = pd.read_csv("output_file.csv", usecols=[6, 5], names=['y', 'ds'], header=None)
        print(df)
        df["ds"] = pd.to_datetime(df["ds"], dayfirst=True)
        df['date'] = df.ds.dt.date
        dfx = pandas.DataFrame(a)
        dfx['date'] = dfx.date.astype('datetime64[ns]')
        df['date'] = df.date.astype('datetime64[ns]')
        df = df.merge(right=dfx, on='date', how='left')

        df = df.sort_values(by="ds")
        df = df.reset_index()
        df = df.drop(labels='index', axis=1)
        df = df.drop(labels='date', axis=1)

        x_train = df.drop(labels='y', axis=1)
        y_train = df.drop(labels='x', axis=1)
        x_train = x_train.reset_index().drop(labels='index', axis=1)
        y_train = y_train.reset_index().drop(labels='index', axis=1)
        x_test = df.drop(labels='y', axis=1)
        y_test = df.drop(labels='x', axis=1)
        x_test = x_test.reset_index().drop(labels='index', axis=1)
        y_test = y_test.reset_index().drop(labels='index', axis=1)
        y_test['y_hat_naive2'] = y_test['y'] + y_test['y'] * (random() / 100)
        y_train['unique_id'] = "BTC"
        x_train['unique_id'] = "BTC"
        y_test['unique_id'] = "BTC"
        x_test['unique_id'] = "BTC"
        limit = statistics.mean(abs(y_train['y'].pct_change().iloc[1:-1].values)) + statistics.stdev(
            abs(y_train['y'].pct_change().iloc[1:-1].values)) * 0.25
        print(x_train)
        print(y_train)
        print(x_test)
        print(y_test)


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



        model = ESRNN(**constants.optimized_hyper_params_for_comb_the_one_i_just_tried)

        # Fit model
        # If y_test_df is provided the model
        # will evaluate predictions on
        # this set every freq_test epochs
        model.fit(x_train, y_train, x_test, y_test, verbose=False)
        listOfDates = []

        date = x_train['ds'].iloc[-1]

        for i in range(5):
            date += timedelta(hours=1)
            listOfDates.append(date)

        data = [{"ds": i,
                "x": int(df.iloc[-1]['x']),
                "unique_id": "BTC"} for i in listOfDates]
        y_hat_df = (model.predict(pd.DataFrame(data)))
        y_hat_df = y_hat_df.dropna()
        y_hat_df = y_hat_df.reset_index().drop(labels='index', axis=1)

        print(y_hat_df)
        # print(smape(np.array([y_train['y'][:-1]]), np.array([y_hat_df['y_hat'].iloc[0]])))
        ai_price = y_hat_df['y_hat'][0]
        real_price_before = \
            pd.read_csv("output_file.csv", usecols=[5], names=['y'], header=None)['y'].iloc[-1]
        print(real_price_before)
        print(limit)
        current_time = pd.read_csv("output_file.csv", usecols=[6], names=['ds'], header=None)['ds'].iloc[-1]
        ct_after = time.time() + 3535
        try:
            avg_funding_rate = get_average_funding_rate(current_time.replace(" ", "T") + "Z")
            print(avg_funding_rate)
        except:
            time.sleep(20)
            avg_funding_rate = get_average_funding_rate(current_time.replace(" ", "T") + "Z")
        if (abs(real_price_before - ai_price) / real_price_before) > limit:
            state = 1
            if state != old_state:
                state_to_action(state)
                trades += 1
                old_state = state
            try:
                time.sleep(ct_after - time.time())
            except:
                pass
        elif real_price_before < ai_price and avg_funding_rate > 0:
            state = 1
            if state != old_state:
                trades += 1
                state_to_action(state)
                old_state = state
                stop_triggered, exit_price = monitor_position(
                    EXCHANGE_INSTANCES['bitstamp'],
                    symbol="BTC/USD",
                    entry_price=real_price_before,
                    position_type='short' if real_price_before > ai_price else 'long',
                    stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                    start_time=datetime.now()
                )
                print(stop_triggered)
                print(exit_price)
                if stop_triggered:
                    state = 0
                    if state != old_state:
                        state_to_action(state)
                        trades += 1
                        old_state = state
                    stop_trigger_count += 1
                try:
                    time.sleep(ct_after - time.time())
                except:
                    pass
            else:
                stop_triggered, exit_price = monitor_position(
                    EXCHANGE_INSTANCES['bitstamp'],
                    symbol="BTC/USD",
                    entry_price=real_price_before,
                    position_type='short' if real_price_before > ai_price else 'long',
                    stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                    start_time=datetime.now()
                )
                if stop_triggered:
                    state = 0
                    if state != old_state:
                        state_to_action(state)
                        trades += 1
                        old_state = state
                    stop_trigger_count += 1
                try:
                    time.sleep(ct_after - time.time())
                except:
                    pass
        elif real_price_before > ai_price and avg_funding_rate < 0:
            state = -1
            if state != old_state:
                state_to_action(state)
                trades += 1
                old_state = state
                stop_triggered, exit_price = monitor_position(
                    EXCHANGE_INSTANCES['bitstamp'],
                    symbol="BTC/USD",
                    entry_price=real_price_before,
                    position_type='short' if real_price_before > ai_price else 'long',
                    stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                    start_time=datetime.now()
                )
                print(stop_triggered)
                print(exit_price)
                if stop_triggered:
                    state = 0
                    if state != old_state:
                        state_to_action(state)
                        trades += 1
                        old_state = state
                    stop_trigger_count += 1
                try:
                    time.sleep(ct_after - time.time())
                except:
                    pass
            else:
                stop_triggered, exit_price = monitor_position(
                    EXCHANGE_INSTANCES['bitstamp'],
                    symbol="BTC/USD",
                    entry_price=real_price_before,
                    position_type='short' if real_price_before > ai_price else 'long',
                    stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                    start_time=datetime.now()
                )
                if stop_triggered:
                    state = 0
                    if state != old_state:
                        state_to_action(state)
                        trades += 1
                        old_state = state
                    stop_trigger_count += 1
            try:
                time.sleep(ct_after - time.time())
            except:
                pass
        else:
            if state == 0:
                try:
                    time.sleep(ct_after - time.time())
                except:
                    pass
            elif state == 1:
                try:
                    time.sleep(ct_after - time.time())
                except:
                    pass
            elif state == -1:
                try:
                    time.sleep(ct_after - time.time())
                except:
                    pass
        predictions += 1
        print(predictions)
        print(stop_trigger_count)
        print(trades)

#from tradingfeatures import bitstamp

#bitstamp = bitstamp()

#bitstamp.update('bitstamp.csv')

run('cryptocom', 'BTC/USD')
