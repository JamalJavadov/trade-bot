from src.core import binance_data
import json

def diagnose():
    print("--- Diagnostic: Futures Positions ---")
    positions = binance_data.get_open_positions()
    print(f"Found {len(positions)} active positions.")
    for p in positions:
        print(f"- {p['symbol']} ({p['side']}): Amt={p['position_amt']}, PnL={p['unrealized_profit']}")
    
    print("\n--- Diagnostic: Open Orders ---")
    orders = binance_data.get_open_orders()
    print(f"Found {len(orders)} open orders.")
    for o in orders:
        print(f"- {o['symbol']} ({o['side']}): Type={o['type']}, Status={o['status']}")

if __name__ == "__main__":
    diagnose()
