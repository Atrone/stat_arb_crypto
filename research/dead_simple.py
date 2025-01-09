import ccxt
import time
import datetime
import certifi
import requests
import datetime



def get_average_funding_rate(max_time):
    """
    Fetches the average funding rate for 60 minutes before the max_time inputted.

    Parameters:
        max_time (str): The maximum timestamp in ISO 8601 format (e.g., '2024-11-22T15:00:00Z').

    Returns:
        float: The average funding rate for the last 60 minutes before max_time.
    """
    # Base URL for the API endpoint
    base_url = "https://archive.prod.vertexprotocol.com/v1"

    # Convert max_time to a datetime object and calculate min_time
    max_time_dt = datetime.datetime.fromisoformat(max_time.replace("Z", "+00:00"))

    # Convert times to Unix timestamp
    max_time_unix = int(max_time_dt.timestamp())
    
    # Prepare the request body (as JSON)
    body = {
        "market_snapshots": {
            "interval": {
                "count": 4,  # We need at least 2 snapshots to get the rate
                "granularity": 7200,  # 1 hour in seconds
                "max_time": max_time_unix
            },
            "product_ids": [2]  # BTC product ID
        }
    }

    # Make the API request
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    response = requests.post(base_url, json=body, headers=headers, verify=False)

    # Check for a successful response
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")

    # Parse the JSON response
    data = response.json()
    
    # Extract funding rates from the snapshots
    funding_rates = []
    if 'snapshots' in data:
        for snapshot in data['snapshots']:
            if 'funding_rates' in snapshot and '2' in snapshot['funding_rates']:
                print(snapshot['funding_rates']['2'])
                funding_rates.append(float(snapshot['funding_rates']['2']))

    if not funding_rates:
        raise ValueError("No funding rate data available for the specified time range.")

    # Calculate the average funding rate
    average_funding_rate = sum(funding_rates) / len(funding_rates)

    return average_funding_rate

# Example usage
if __name__ == "__main__":
    exchange = ccxt.vertex()
    exchange.load_markets()
    print(exchange.fetch_funding_rate("BTC/USDC:USDC"))
    max_time_input = "2024-11-23T22:00:00Z"
    try:
        avg_rate = get_average_funding_rate(max_time_input)
        print(f"Average funding rate for the 60 minutes before {max_time_input}: {avg_rate}")
    except Exception as e:
        print(f"Error: {e}")