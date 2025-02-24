import requests
from datetime import datetime, timedelta
import time
from typing import Dict

def analyze_funding_rate_switches(start_time=None) -> Dict:
    """
    Analyzes how often the funding rate switches between positive and negative
    over the past month.
    
    Args:
        start_time: Optional datetime to start analysis from. Defaults to current time.
    
    Returns:
        dict: Statistics about funding rate switches including:
            - total_switches: number of times rate changed sign
            - avg_switch_duration: average duration between switches (in hours)
            - positive_periods: number of positive rate periods
            - negative_periods: number of negative rate periods
            - avg_positive_rate: average positive rate
            - avg_negative_rate: average negative rate
    """
    base_url = "https://archive.prod.vertexprotocol.com/v1"
    
    if not start_time:
        start_time = datetime.utcnow()
    
    # Get data for past month (30 days)
    # Using 3000 second intervals (50 minutes) to get good granularity
    intervals_needed = int((300 * 24 * 3600) / 3000)  # 3 days worth of 50-minute intervals
    
    funding_rates = []
    current_time = start_time
    
    # Fetch data in chunks to avoid too large requests
    chunk_size = 100  # Number of intervals per request
    for _ in range(0, intervals_needed, chunk_size):
        time.sleep(3)  # Rate limiting
        
        max_time_unix = int(current_time.timestamp())
        
        body = {
            "market_snapshots": {
                "interval": {
                    "count": min(chunk_size, intervals_needed - len(funding_rates)),
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
        
        try:
            response = requests.post(base_url, json=body, headers=headers, verify=False)
            if response.status_code == 200:
                data = response.json()
                if 'snapshots' in data:
                    for snapshot in data['snapshots']:
                        if 'funding_rates' in snapshot and '2' in snapshot['funding_rates']:
                            timestamp = int(snapshot['timestamp'])
                            rate = float(snapshot['funding_rates']['2'])
                            funding_rates.append((timestamp, rate))
            
            # Update current_time for next iteration
            current_time = datetime.fromtimestamp(min([rate[0] for rate in funding_rates]))
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            continue

    # Sort by timestamp
    funding_rates.sort(key=lambda x: x[0])
    
    # Analysis variables
    total_switches = 0
    switch_durations = []
    positive_rates = []
    negative_rates = []
    rates = []
    last_sign = None
    last_switch_time = None
    
    # Analyze the data
    for timestamp, rate in funding_rates:
        current_sign = 1 if rate > 0 else -1
        
        if rate > 0:
            positive_rates.append(rate)
        else:
            negative_rates.append(rate)
        rates.append(rate)
        if last_sign is not None and current_sign != last_sign:
            total_switches += 1
            if last_switch_time is not None:
                duration = (timestamp - last_switch_time) / 3600  # Convert to hours
                switch_durations.append(duration)
            last_switch_time = timestamp
            
        last_sign = current_sign
    
    return {
        "total_switches": total_switches,
        "avg_switch_duration": sum(switch_durations) / len(switch_durations) if switch_durations else 0,
        "positive_periods": len(positive_rates),
        "negative_periods": len(negative_rates),
        "avg_positive_rate": sum(positive_rates) / len(positive_rates) if positive_rates else 0,
        "avg_negative_rate": sum(negative_rates) / len(negative_rates) if negative_rates else 0,
        "std_positive_rate": (sum((x - (sum(positive_rates) / len(positive_rates))) ** 2 for x in positive_rates) / len(positive_rates)) ** 0.5 if positive_rates else 0,
        "std_negative_rate": (sum((x - (sum(negative_rates) / len(negative_rates))) ** 2 for x in negative_rates) / len(negative_rates)) ** 0.5 if negative_rates else 0,
        "avg_rate": sum(rates) / len(rates) if rates else 0,
        "std_rate": (sum((x - (sum(rates) / len(rates))) ** 2 for x in rates) / len(rates)) ** 0.5 if rates else 0,
        "total_samples": len(funding_rates)
    }

if __name__ == "__main__":
    stats = analyze_funding_rate_switches()
    print("\nFunding Rate Analysis:")
    print(f"Total number of sign switches: {stats['total_switches']}")
    print(f"Average duration between switches: {stats['avg_switch_duration']:.2f} hours")
    print(f"Number of positive rate periods: {stats['positive_periods']}")
    print(f"Number of negative rate periods: {stats['negative_periods']}")
    print(f"Average positive rate: {stats['avg_positive_rate']:.6f}")
    print(f"Average negative rate: {stats['avg_negative_rate']:.6f}")
    print(f"Standard deviation of positive rate: {stats['std_positive_rate']:.6f}")
    print(f"Standard deviation of negative rate: {stats['std_negative_rate']:.6f}")
    print(f"Total samples analyzed: {stats['total_samples']}")
    print(f"Average rate: {stats['avg_rate']:.6f}")
    print(f"Standard deviation of rate: {stats['std_rate']:.6f}")
    print(-600077852205 - 1620712129010 * 3)
    print(11616242444237 + 5250911279712 * 3)
11006365253631.535156
27368976283373
4262058534825
5250911279712.157227
1020634276805
2641346405815
-600077852205.416626
-5462214239235
1620712129010.400391