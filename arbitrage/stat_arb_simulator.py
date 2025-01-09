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


def simulate(exchange, symbol):
    fear_greed = requests.get('https://api.alternative.me/fng/?limit=0&format=csv').text.split("\n", 3)[3].split('\n')[:-2]
    a = []

    for fg in fear_greed[1:-3]:
        a.append({"date": datetime.strptime(fg.split(",")[0], "%d-%m-%Y"), "x": int(fg.split(",")[1])})

    stop_loss_percentage = 0.005
    stop_loss_percentage_long = 0.01

    print(a)

    import pandas as pd

    df = pd.read_csv("Bitstamp_BTCUSD_1h_4.csv", usecols=[1, 3], names=['ds', 'y'], header=None).iloc[::-1]
    print(df)
    df["ds"] = pd.to_datetime(df["ds"], dayfirst=True)
    df['date'] = df.ds.dt.date
    dfx = pandas.DataFrame(a)
    dfx['date'] = dfx.date.astype('datetime64[ns]')
    df['date'] = df.date.astype('datetime64[ns]')
    df = df.merge(right=dfx, on='date', how='left')

    df = df.sort_values(by="ds")
    df = df.reset_index()
    df = df.iloc[:-2]
    df = df.drop(labels='index', axis=1)
    df = df.drop(labels='date', axis=1)
    smapes = []
    successes = 0
    failures = 0
    predictions = 0
    money = 0
    successes_list = []
    failures_list = []
    trades = 0
    old_state = 0
    state = 0
    stop_trigger_count = 0
    end = 1000 # changed from 1000 2024/11/06
    # THIS IS 6 STOP LOSS TRIGGERS (MAX )
    for n in range(150, 250):  # 1-45 GOOD (+3100), 45-57 GOOD (-400), 57-83 GOOD (-300), 85-100 GOOD (-350)
        # (2100 PROFIT ON 1 BTC over 12.5 days)
        # 120-140 GOOD (+200)
        # 140-160 GOOD (+700)
        # 160-200 GOOD (-800)
        # (2200 PROFIT ON 1 BTC over 25 days)
        # NEW
        # 200-500 GOOD (+3300)
        # NEW
        # (3300 PROFIT ON 1 BTC over 30 days)
        # 500-540 GOOD (+650)
        # 550-580 GOOD (+500)
        # 580-680 GOOD (+700)
        # 700-720 GOOD (-400)
        # 720-740 GOOD (+600)
        # 740-770 GOOD (+200)
        # (2250 PROFIT ON 1 BTC over 30 days)
        # ABOVE IS FOR _1

        # HERE IS _3:
        # 1-250 (+4000) # actually +3136
        # (4000 PROFIT ON 1 BTC over 25 days)
        # 250-275 (+300) # actually +255
        # 275-300 (-400)
        # 300-335 (-1900)
        # 335-360 (+600)
        # 360-380 (-900)
        # 380-425 (-2000)
        # 425-500 # actually +2335
        # 54% WIN RATE
        # KEEP TEST LOSS < 0.002



        # THIS IS limit=0.25, both stop losses=150 @ 0.005 short, 0.01 long, comb hypers:
        # BELOW THIS IS _3 csv
        # 1-250 (+4000) # actually +3136 # actually -400 (no short) # actually -50 (no short, old params) # actually 979.8090280744739 (no short, old params, no /2)
        # # 28
        # # 0.003950443516706758
        # # 247.0347545064251
        # # 168
        # # 93 (cursor MUST RETUNE ON _3 and _1 (make sure _4 still works))
        # 250-275 (+300) # actually +255
        # 275-300 (-400) # actually +67
        # 300-335 (-1900) # actually -558
        # 335-360 (+600) # actually +510
        # 360-380 (-900) # actually -742
        # 380-425 (-2000) # actually -2000
        # 425-500 # actually +2335
        # ABOVE THIS IS _3 csv
        # BELOW THIS IS _1 csv
        # 45-57 GOOD (+100) # actually +100 # 1-57 actually -850 (cursor) # actually (no short) -200.779563365448 (early stop at ~45) # actually -23.71162416168788 (no short)
        # 38
        # 7
        # 13
        # 1-45 GOOD (+258) # actually +258
        # ABOVE THIS IS _1 csv
        # BELOW THIS IS _4 csv
        # 1-45 GOOD (+2122) # actually +2122
        # 45-100 GOOD (+550) # actually +550 # actually -1400 (cursor (just did it today 11/19 and i did 1-100))
        # 100-150 GOOD (+1576) # actually +1576
        # 150-250 GOOD (-4000) # actually -3800 # actually -5818 (cursor (just did it today 11/20 and i did 100-250))
        # 250-350 GOOD (+4900) # actually +4900
        # ABOVE THIS IS _4 csv
        # STILL AROUND 54% WIN RATE
        # ~1000 samples
        # BELOW THIS IS _5 csv
        # 800-900 GOOD (+3900) # actually +3900 # actually +2500 (cursor) # actually +4652 (very low short stop-loss) # actually +900 (no short) # actually -369.01643381947594 (no short, old params, no /2) # actually +840 (perp futures addon 11/23)
        # 700-800 GOOD (+2800) # actually +2800 # actually +1722 (cursor) # actually +1866.0749741445347 (no short) # actually +1472.8172605049076 (no short, old params, no /2)
        # 29
        # 14
        # 600-700 GOOD (+3330) # actually -1300 (no short) # actually -277 (no short, old params, no /2) # -1353.5936478205615
        # ABOVE THIS IS _5 csv
        # 54% WIN RATE
        # KEEP TEST LOSS < 0.002
        # KEEP TEST LOSS < 0.001 for each csv (cursor)


        # this is the i just tried params, funding rate, limit = 0.25, short loss 0.5%, long loss 1%
        # _3
        # (1-250) +2381.7747742142883
        # (250-350) -886.2146842064366
        # (360-460) -1000.00
        # _4 
        #+2044.2080660286929 (1-100)
        #80
        #24
        #14
        # +6000.00 (250-350)
        # _5
        # -1350.00 (600-700)
        # +967.7458176597763 (700-800)
        # -207.2085659829528 (800-900)
        # -1462.2612239729012 (600-700)
        # _1
        # 1-50 +717.00
        # 50-150 +115.59741495211688
        # _6
        # 5
        # 0.02288168562463909
        # 15
        # 12393.42631087813
        # 191
        # 44
        # 35


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
                "x": int(df.iloc[n]['x']),
                "unique_id": "BTC"} for i in listOfDates]
        y_hat_df = (model.predict(pd.DataFrame(data)))
        y_hat_df = y_hat_df.dropna()
        y_hat_df = y_hat_df.reset_index().drop(labels='index', axis=1)

        print(y_hat_df)
        # print(smape(np.array([y_train['y'][:-1]]), np.array([y_hat_df['y_hat'].iloc[0]])))
        ai_price = y_hat_df['y_hat'][0]
        print(pd.read_csv("Bitstamp_BTCUSD_1h_4.csv", usecols=[1], names=['y'], header=None)['y'].iloc[::-1].iloc[
                -(end - n + 3)])
        print(pd.read_csv("Bitstamp_BTCUSD_1h_4.csv", usecols=[1], names=['y'], header=None)['y'].iloc[::-1].iloc[
                -(end - n + 2)])
        real_price_before = \
            pd.read_csv("Bitstamp_BTCUSD_1h_4.csv", usecols=[3], names=['y'], header=None)['y'].iloc[::-1].iloc[
                -(end - n + 3)]
        real_price_after = \
            pd.read_csv("Bitstamp_BTCUSD_1h_4.csv", usecols=[3], names=['y'], header=None)['y'].iloc[::-1].iloc[
                -(end - n + 2)]
        print(real_price_before)
        print(real_price_after)
        print(limit)
        # When initializing:
        monitor_created = create_price_monitor(EXCHANGE_INSTANCES['bitstamp'])

        if (abs(real_price_before - ai_price) / real_price_before) > limit:
            state = 1
            if state != old_state:
                trades += 1
                old_state = state
            money += (real_price_after - real_price_before)
            print(smape(ai_price, real_price_after))
            smapes.append(smape(ai_price, real_price_after))
            print(money)
            print(predictions)
            print(successes)
            print(failures)
            print(successes_list)
            print(failures_list)
            print(smapes)
            continue
        fees = get_exchange_fees(EXCHANGE_INSTANCES[exchange])[0] / 3
        slippage = estimate_slippage(EXCHANGE_INSTANCES[exchange], 'BTC/USDC:USDC', real_price_before, 'buy') / 3
        print(slippage)
        print(fees)
        current_time = pd.read_csv("Bitstamp_BTCUSD_1h_4.csv", usecols=[1], names=['ds'], header=None)['ds'].iloc[::-1].iloc[
        -(end - n + 3)]
        avg_funding_rate = get_average_funding_rate(current_time.replace(" ", "T") + "Z")
        print(avg_funding_rate)
        if real_price_before < ai_price and avg_funding_rate > 0:
            state = 1
            if state != old_state:
                trades += 1
                old_state = state
            if real_price_before < real_price_after:
                money += abs(real_price_after - real_price_before) - real_price_before * fees - real_price_before * slippage
                successes += 1
                failures_list.append(0)
                successes_list.append(1)
            else:
                stop_triggered, exit_price = monitor_position(
                    EXCHANGE_INSTANCES['bitstamp'],
                    symbol="BTC/USD",
                    entry_price=real_price_before,
                    position_type='short' if real_price_before > ai_price else 'long',
                    stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                    current_time=current_time
                )
                print(stop_triggered)
                print(exit_price)
                if stop_triggered:
                    state = 0
                    if state != old_state:
                        trades += 1
                        old_state = state
                    stop_trigger_count += 1
                    failures += 1
                    money -= abs(exit_price - real_price_before) + real_price_before * fees + real_price_before * slippage
                    continue
                money -= abs(real_price_after - real_price_before) + real_price_before * fees + real_price_before * slippage
                failures += 1
                failures_list.append(1)
                successes_list.append(0)
        elif real_price_before > ai_price and avg_funding_rate < 0:
            state = -1
            if state != old_state:
                trades += 1
                old_state = state
            if real_price_before > real_price_after:
                money += abs(real_price_after - real_price_before) - real_price_before * fees - real_price_before * slippage
                successes += 1
                failures_list.append(0)
                successes_list.append(1)
            else:
                stop_triggered, exit_price = monitor_position(
                    EXCHANGE_INSTANCES['bitstamp'],
                    symbol="BTC/USD",
                    entry_price=real_price_before,
                    position_type='short' if real_price_before > ai_price else 'long',
                    stop_loss_pct=stop_loss_percentage if real_price_before > ai_price else stop_loss_percentage_long,
                    current_time=current_time
                )
                print(stop_triggered)
                print(exit_price)
                if stop_triggered:
                    state = 0
                    if state != old_state:
                        trades += 1
                        old_state = state
                    stop_trigger_count += 1
                    failures += 1
                    money -= abs(exit_price - real_price_before) + real_price_before * fees + real_price_before * slippage
                    continue
                money -= abs(real_price_after - real_price_before) + real_price_before * fees + real_price_before * slippage
                failures += 1
                failures_list.append(1)
                successes_list.append(0)
        else:
            if state == 0:
                continue
            elif state == 1:
                money += (real_price_after - real_price_before)
            elif state == -1:
                money -= (real_price_after - real_price_before)
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


simulate('vertex', 'BTC/USDC:USDC')
