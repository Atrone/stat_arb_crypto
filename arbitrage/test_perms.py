import time

import cloudscraper

BASE_URL = ""
# Replace with your credentials
email = ""
password = ""
broker_id = ""


def login(email, password, broker_id):
    LOGIN_ENDPOINT = f"{BASE_URL}/manager/co-login"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "email": email,
        "password": password,
        "brokerId": broker_id
    }

    # Initialize Cloudscraper session
    scraper = cloudscraper.create_scraper()

    # Make a login request
    response = scraper.post(
        LOGIN_ENDPOINT, json=payload, headers=headers
    )
    if response.status_code == 200:
        print(f"Login successful with broker ID: {broker_id}")
        return response.json()
    else:
        print(f"Login failed with broker ID: {broker_id}")
        print("Error:", response.text)
        return None

def open_btcusd_position(order_side, volume, is_mobile=False):
    """
    Opens a BTCUSD trading position.

    Parameters:
        order_side (str): "BUY" or "SELL".
        volume (float): Amount of the trade.
        trading_api_token (str): Trading API token from the login response.
        auth_token (str): Authentication token from the login response.
        system_uuid (str): System UUID provided by the broker.
        is_mobile (bool): True if request originates from a mobile device; otherwise False.

    Returns:
        dict: Response from the server.
    """
    while True:
        login_data = login(email, password, broker_id)
        if login_data:
            print(login_data)
            auth_token = login_data["token"]
            trading_api_token = login_data["accounts"][0]["tradingApiToken"]
            system_uuid = login_data["accounts"][0]['offer']["system"]['uuid']  # Replace with actual system UUID.
            OPEN_POSITION_ENDPOINT = f"{BASE_URL}/mtr-api/{system_uuid}/position/open"
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Auth-trading-api": trading_api_token,
                "Cookie": f"co-auth={auth_token}"
            }
            payload = {
                "instrument": "BTCUSD",  # Hardcoded for BTCUSD
                "orderSide": order_side,  # "BUY" or "SELL"
                "volume": volume,
                "slPrice": 0,  # Stop Loss price, 0 if not set
                "tpPrice": 0,  # Take Profit price, 0 if not set
                "isMobile": is_mobile
            }
            print(f"Endpoint: {OPEN_POSITION_ENDPOINT}")
            print(f"Payload: {payload}")
            # Initialize Cloudscraper session
            
            scraper = cloudscraper.create_scraper()

            response = scraper.post(OPEN_POSITION_ENDPOINT, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                time.sleep(1)
                print("Open BTCUSD Position Error:", response.text)
        time.sleep(1)

def close_btcusd_positions(is_mobile=False):
    """
    Closes all BTCUSD trading positions.

        is_mobile (bool): True if request originates from a mobile device; otherwise False.

    Returns:
        dict: Response from the server.
    """
    while True:
        login_data = login(email, password, broker_id)
        if login_data:
            print(login_data)
            auth_token = login_data["token"]
            trading_api_token = login_data["accounts"][0]["tradingApiToken"]
            system_uuid = login_data["accounts"][0]['offer']["system"]['uuid']  # Replace with actual system UUID.
            CLOSE_POSITIONS_ENDPOINT = f"{BASE_URL}/mtr-api/{system_uuid}/positions/close"
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Auth-trading-api": trading_api_token,
                "Cookie": f"co-auth={auth_token}"
            }
            # Initialize Cloudscraper session

            positions = opened_positions()
            print(positions)
            for position in positions["positions"]:
                payload = {
                    "instrument": "BTCUSD"
                }
                payload["positionId"] = position["id"]
                payload["orderSide"] = position["side"]
                payload["volume"] = position["volume"]
                while True:
                    scraper = cloudscraper.create_scraper()
                    payload_list = [payload]
                    response = scraper.post(CLOSE_POSITIONS_ENDPOINT, json=payload_list, headers=headers)
                    if response.status_code == 200:
                        print(response.json())
                        break
                    else:
                        time.sleep(1)
                        print("Close BTCUSD Position Error:", response.text)
            return True
        time.sleep(1)



def opened_positions(is_mobile=False):
    """
    """
    while True:
        login_data = login(email, password, broker_id)
        if login_data:
            print(login_data)
            auth_token = login_data["token"]
            trading_api_token = login_data["accounts"][0]["tradingApiToken"]
            system_uuid = login_data["accounts"][0]['offer']["system"]['uuid']  # Replace with actual system UUID.
            OPENED_POSITION_ENDPOINT = f"{BASE_URL}/mtr-api/{system_uuid}/open-positions"
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Auth-trading-api": trading_api_token,
                "Cookie": f"co-auth={auth_token}"
            }
            # Initialize Cloudscraper session
            scraper = cloudscraper.create_scraper()

            response = scraper.get(OPENED_POSITION_ENDPOINT, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                print("Opened Positions Error:", response.text)
                time.sleep(1)
        time.sleep(1)
