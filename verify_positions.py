
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.core import binance_data

def verify():
    print("Checking connection...")
    try:
        ping = binance_data.ping()
        print(f"Ping successful: {ping}")
    except Exception as e:
        print(f"Ping failed: {e}")
        return

    print("\nFetching open positions...")
    try:
        positions = binance_data.get_open_positions()
        print(f"Found {len(positions)} open positions.")
        for p in positions:
            print(p)
    except Exception as e:
        print(f"Failed to fetch positions: {e}")

    print("\nFetching open orders...")
    try:
        orders = binance_data.get_open_orders()
        print(f"Found {len(orders)} open orders.")
        for o in orders:
            print(o)
    except Exception as e:
        print(f"Failed to fetch orders: {e}")

    print("\nDeep checking EDUUSDT...")
    try:
        # Access client directly to bypass filtering
        raw_positions = binance_data.client.futures_position_information(symbol="EDUUSDT")
        print(f"DTO type: {type(raw_positions)}")
        print(f"Raw data: {raw_positions}")
    except Exception as e:
        print(f"Failed to fetch EDUUSDT raw: {e}")

if __name__ == "__main__":
    verify()
