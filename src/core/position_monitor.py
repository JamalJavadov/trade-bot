"""
Deep Position Monitor Module

This module provides a background service that continuously monitors active positions
using professional deep analysis (Technicals + Market Structure + News).
"""
from __future__ import annotations

import threading
import time
import queue
import logging
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from . import binance_data
from . import deep_analyzer
from .news import NewsAnalyzer

class PositionMonitor:
    """
    Monitors active positions in the background.
    
    Features:
    - Periodically fetches active positions.
    - Runs Deep Analysis (Technicals, Structure, Volume, MTF) for each position.
    - Fetches relevant News for specific coins.
    - Calculates a 'Health Score' for the trade.
    - Emit results to a UI queue.
    """
    
    def __init__(self, settings: Dict[str, Any], msg_queue: queue.Queue):
        self.settings = settings
        self.queue = msg_queue
        self.stop_event = threading.Event()
        self.news_analyzer = NewsAnalyzer()
        
        # Concurrency
        self.max_workers = 3
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # State
        self.active_positions: List[Dict[str, Any]] = []
        self.analysis_results: Dict[str, Any] = {}
        self.news_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Metrics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        
        # Configuration
        self.refresh_interval = 60  # seconds
        
    def start(self):
        """Start the monitor thread."""
        self.stop_event.clear()
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def stop(self):
        """Stop the monitor thread."""
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        
    def _monitor_loop(self):
        """Main loop for monitoring positions."""
        while not self.stop_event.is_set():
            try:
                # 1. Fetch Active Positions
                positions = binance_data.get_open_positions()
                self.active_positions = positions
                
                if not positions:
                    self.queue.put(("monitor_status", "No active positions"))
                    time.sleep(5)
                    continue
                    
                self.queue.put(("monitor_status", f"Monitoring {len(positions)} active positions..."))
                
                # 2. Analyze each position in parallel
                futures = []
                for pos in positions:
                    symbol = pos.get("symbol")
                    if symbol:
                        futures.append(self.executor.submit(self._analyze_position_task, symbol, pos))
                
                # Wait for all to complete (or just let them run)
                # For UI responsiveness, we let them emit their own results as they finish
                
                # 3. Sleep until next cycle
                for _ in range(self.refresh_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.queue.put(("monitor_error", str(e)))
                time.sleep(10)
                
    def _analyze_position_task(self, symbol: str, position_data: Dict[str, Any]):
        """Worker task to analyze a single position."""
        start_time = time.time()
        try:
            # A. Deep Technical Analysis
            # We need a fetch_ohlcv wrapper that matches deep_analyzer's expected signature
            analysis_result = deep_analyzer.perform_deep_analysis(
                symbol=symbol,
                fetch_ohlcv=binance_data.get_ohlcv,
                settings=self.settings,
                on_stage=None # No fine-grained progress needed for background monitor
            )
            
            # B. News Analysis (Specific to this coin)
            # We filter news by symbol name (e.g. "BTC", "ETH")
            base_asset = symbol.replace("USDT", "")
            news_items = self._fetch_relevant_news(base_asset)
            
            # C. Synthesize "Health Score"
            health_score, health_reason = self._calculate_health(analysis_result, news_items, position_data)
            
            elapsed = time.time() - start_time
            self.tasks_completed += 1
            self.total_processing_time += elapsed
            avg_time = (self.total_processing_time / self.tasks_completed) * 1000 if self.tasks_completed else 0
            
            # D. Package Result
            result_payload = {
                "symbol": symbol,
                "position": position_data,
                "deep_analysis": analysis_result,
                "news": news_items,
                "health": {
                    "score": health_score,
                    "reason": health_reason,
                    "sentiment": "BULLISH" if health_score > 60 else "BEARISH" if health_score < 40 else "NEUTRAL"
                },
                "metrics": {
                    "tasks_completed": self.tasks_completed,
                    "tasks_failed": self.tasks_failed,
                    "avg_time_ms": avg_time,
                    "active_threads": self.max_workers # approximated
                },
                "timestamp": time.time()
            }
            
            # E. Emit to UI
            self.queue.put(("position_analysis_update", result_payload))
            
        except Exception as e:
            self.tasks_failed += 1
            print(f"Error analyzing {symbol}: {e}")

    def _fetch_relevant_news(self, coin: str) -> List[Dict[str, Any]]:
        """Fetch and filter news for a specific coin."""
        # For efficiency, we might want to fetch global news once and filter, 
        # but for now we'll fetch latest and filter properties.
        # Note: NewsAnalyzer currently fetches global news. 
        # We can reuse the cache or fetch fresh.
        try:
            all_news = self.news_analyzer.get_latest_news(limit=20)
            
            relevant = []
            for item in all_news:
                # Simple keyword matching
                text_content = (item['title'] + " " + item['description']).lower()
                if coin.lower() in text_content:
                    relevant.append(item)
            
            # If no specific news, return top global news (context is important)
            if not relevant:
                return all_news[:3]
                
            return relevant
        except Exception:
            return []

    def _calculate_health(self, 
                         analysis: deep_analyzer.DeepAnalysisResult, 
                         news: List[Dict[str, Any]], 
                         pos: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate a 0-100 score representing the health of the trade.
        """
        score = 50.0
        reasons = []
        
        # 1. Technical Alignment (Max 40 pts)
        # If we are LONG, and signal is BUY -> Good (+40)
        # If we are LONG, and signal is SELL -> Bad (-40)
        side = pos.get("side", "LONG")
        tech_signal = analysis.signal
        
        if side == "LONG":
            if "BUY" in tech_signal:
                score += 30
                reasons.append("Technicals Support Trade")
            elif "SELL" in tech_signal:
                score -= 30
                reasons.append("Technicals Oppose Trade")
        else: # SHORT
            if "SELL" in tech_signal:
                score += 30
                reasons.append("Technicals Support Trade")
            elif "BUY" in tech_signal:
                score -= 30
                reasons.append("Technicals Oppose Trade")
                
        # 2. PnL Check (Max 30 pts)
        roi = 0.0
        try:
            pnl = float(pos.get("unrealized_profit", 0))
            margin = float(pos.get("initialMargin", 1)) # avoid div/0
            roi = (pnl / margin) * 100 if margin else 0
        except:
            pass
            
        if roi > 0:
            score += min(20, roi) # Cap at +20
        else:
            score -= min(20, abs(roi)) # Cap at -20
            
        # 3. News Sentiment (Max 30 pts)
        bullish_news = sum(1 for n in news if n['sentiment']['label'] == 'POSITIVE')
        bearish_news = sum(1 for n in news if n['sentiment']['label'] == 'NEGATIVE')
        
        if side == "LONG":
            if bullish_news > bearish_news:
                score += 20
                reasons.append("Positive News Sentiment")
            elif bearish_news > bullish_news:
                score -= 20
                reasons.append("Negative News Sentiment")
        else:
             if bearish_news > bullish_news:
                score += 20
                reasons.append("Negative News Sentiment (Good for Short)")
             elif bullish_news > bearish_news:
                score -= 20
                reasons.append("Positive News Sentiment (Bad for Short)")
        
        # Clamp
        score = max(0.0, min(100.0, score))
        
        reason_str = ", ".join(reasons) if reasons else "Neutral conditions"
        return score, reason_str
