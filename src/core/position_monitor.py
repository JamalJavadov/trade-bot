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
                # 1. Fetch Active Positions & Open Orders
                positions = binance_data.get_open_positions()
                orders = binance_data.get_open_orders()
                
                self.active_positions = positions
                
                if not positions and not orders:
                    self.queue.put(("monitor_status", "No active positions or open orders"))
                    self.queue.put(("monitor_update_clear", {})) # Clear dashboard if empty
                    time.sleep(10)
                    continue
                    
                self.queue.put(("monitor_status", f"Monitoring {len(positions)} positions & {len(orders)} orders..."))
                print(f"DEBUG: Monitoring Positions: {[p['symbol'] for p in positions]}")
                print(f"DEBUG: Monitoring Orders: {[o['symbol'] for o in orders]}")
                
                # 2. Analyze each in parallel
                futures = []
                
                # Positions
                for pos in positions:
                    symbol = pos.get("symbol")
                    if symbol:
                        futures.append(self.executor.submit(self._analyze_position_task, symbol, pos, "POSITION"))
                
                # Orders
                for ord in orders:
                    symbol = ord.get("symbol")
                    if symbol:
                        # Skip if we already analyze this coin via Position (to avoid redundancy)
                        if any(p.get("symbol") == symbol for p in positions):
                            continue
                        futures.append(self.executor.submit(self._analyze_position_task, symbol, ord, "ORDER"))
                
                # Wait for next cycle
                for _ in range(self.refresh_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.queue.put(("monitor_error", str(e)))
                time.sleep(10)
                
    def _analyze_position_task(self, symbol: str, data: Dict[str, Any], type: str):
        """Worker task to analyze a single account item (Position or Order)."""
        start_time = time.time()
        try:
            # A. Deep Technical Analysis
            analysis_result = deep_analyzer.perform_deep_analysis(
                symbol=symbol,
                fetch_ohlcv=binance_data.get_ohlcv,
                settings=self.settings,
                on_stage=None
            )
            
            # B. News Analysis
            base_asset = symbol.replace("USDT", "")
            news_items = self._fetch_relevant_news(base_asset)
            
            # C. Synthesize "Health Score"
            health_score, health_reason = self._calculate_health(analysis_result, news_items, data, type)
            
            elapsed = time.time() - start_time
            self.tasks_completed += 1
            self.total_processing_time += elapsed
            avg_time = (self.total_processing_time / self.tasks_completed) * 1000 if self.tasks_completed else 0
            
            # D. Package Result
            result_payload = {
                "symbol": symbol,
                "type": type,
                "data": data,
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
                    "active_threads": self.max_workers
                },
                "timestamp": time.time()
            }
            
            # E. Emit to UI
            self.queue.put(("position_analysis_update", result_payload))
            
        except Exception as e:
            self.tasks_failed += 1
            print(f"Error analyzing {symbol}: {e}")
            
            # Emit a failure status to the UI so the asset still appears in sidebar
            fail_payload = {
                "symbol": symbol,
                "type": type,
                "status": "FAILED",
                "confidence": 0.0,
                "signal": "ERROR",
                "reasons": [f"Analysis Error: {str(e)}"],
                "health": {"score": 0, "reason": "System Error"}
            }
            self.queue.put(("position_analysis_update", fail_payload))

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
                         data: Dict[str, Any],
                         type: str) -> Tuple[float, str]:
        """
        Calculate health score. Works for both existing Positions and Pending Orders.
        """
        score = 50.0
        reasons = []
        
        # Determine side
        side = data.get("side", "LONG")
        if type == "ORDER":
            # For orders, side might be BUY/SELL. Map to LONG/SHORT
            side = "LONG" if side == "BUY" else "SHORT"
            
        tech_signal = analysis.signal
        
        # 1. Technical Alignment
        if side == "LONG":
            if "BUY" in tech_signal:
                score += 30
                reasons.append("Trend Supports Entry")
            elif "SELL" in tech_signal:
                score -= 30
                reasons.append("Trend Invalidation")
        else: # SHORT
            if "SELL" in tech_signal:
                score += 30
                reasons.append("Trend Supports Entry")
            elif "BUY" in tech_signal:
                score -= 30
                reasons.append("Trend Invalidation")
                
        # 2. Performance/Price Check
        if type == "POSITION":
            try:
                pnl = float(data.get("unrealized_profit", 0))
                margin = float(data.get("initial_margin", 1))
                roi = (pnl / margin) * 100 if margin else 0
                if roi > 0: score += min(15, roi/2)
                else: score -= min(25, abs(roi))
            except: pass
        else: # ORDER
            # Is price moving away or towards entry?
            try:
                entry = float(data.get("price", 0))
                current = analysis.entry # current price proxy from analyzer
                # If current price is far from entry, validity drops
                dist = abs(current - entry) / entry
                if dist > 0.05: # >5% away
                    score -= 20
                    reasons.append("Price far from Limit")
            except: pass
            
        # 3. News
        bullish_news = sum(1 for n in news if n['sentiment']['label'] == 'POSITIVE')
        bearish_news = sum(1 for n in news if n['sentiment']['label'] == 'NEGATIVE')
        
        if (side == "LONG" and bullish_news > bearish_news) or (side == "SHORT" and bearish_news > bullish_news):
            score += 10
            reasons.append("News Confluence")
        
        score = max(0.0, min(100.0, score))
        return score, ", ".join(reasons) if reasons else "Evaluating..."
