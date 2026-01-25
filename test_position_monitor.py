
import sys
import os
import time
import queue
import threading
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from src.core.position_monitor import PositionMonitor

def test_monitor():
    print("Initializing PositionMonitor...")
    
    # Mock settings
    settings = {
        "deep_analysis": {},
        "advanced_scoring": {},
        "strategy": {"impulse_timeframe": "1h"}
    }
    msg_queue = queue.Queue()
    
    monitor = PositionMonitor(settings, msg_queue)
    monitor.refresh_interval = 2 # Speed up for test
    
    # Mock Binance Data
    mock_positions = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.01",
            "entryPrice": "50000",
            "markPrice": "51000",
            "unrealizedProfit": "10.0",
            "side": "LONG"
        }
    ]
    
    # We need to mock binance_data.get_open_positions AND deep_analyzer.perform_deep_analysis
    # because perform_deep_analysis calls fetch_ohlcv which calls API.
    
    with patch("src.core.position_monitor.binance_data.get_open_positions", return_value=mock_positions), \
         patch("src.core.position_monitor.deep_analyzer.perform_deep_analysis") as mock_analyze, \
         patch("src.core.position_monitor.NewsAnalyzer.get_latest_news") as mock_news:
        
        # Setup mock analysis result
        mock_result = MagicMock()
        mock_result.symbol = "BTCUSDT"
        mock_result.signal = "STRONG_BUY"
        mock_result.confidence = 85.0
        mock_result.quality_score = 90.0
        mock_analyze.return_value = mock_result
        
        # Setup mock news
        mock_news.return_value = [
            {"title": "BTC hits 100k", "description": "Market is flying", "sentiment": {"label": "POSITIVE"}}
        ]
        
        print("Starting Monitor...")
        monitor.start()
        
        print("Waiting for updates...")
        start_time = time.time()
        updates_received = 0
        
        try:
            while time.time() - start_time < 10:
                try:
                    kind, payload = msg_queue.get(timeout=1)
                    if kind == "position_analysis_update":
                        updates_received += 1
                        print(f"\n[UPDATE RECEIVED] Symbol: {payload['symbol']}")
                        print(f"Health Score: {payload['health']['score']}")
                        print(f"Deep Analysis Confidence: {payload['deep_analysis'].confidence}")
                        print(f"News Items: {len(payload['news'])}")
                        
                        if updates_received >= 1:
                            print("\nSUCCESS: Received analysis update!")
                            break
                    elif kind == "monitor_status":
                        print(f"[STATUS] {payload}")
                except queue.Empty:
                    continue
        finally:
            monitor.stop()
            print("Monitor Stopped.")

if __name__ == "__main__":
    test_monitor()
