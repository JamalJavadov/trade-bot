"""
Professional Parallel Analysis Engine

This module provides high-performance parallel analysis capabilities
that leverage multithreading for professional-grade market analysis.
It integrates all analysis modules with concurrent execution.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .thread_manager import (
    get_thread_manager,
    ThreadPoolConfig,
    ProfessionalThreadManager,
    ParallelDataFetcher,
    ThreadSafeDict,
    ThreadSafeList,
    TaskResult,
)

# Import analysis modules
from .indicators import calculate_all_indicators, IndicatorSummary
from .market_structure import analyze_market_structure, MarketStructure
from .volume_analysis import analyze_volume, VolumeAnalysis
from .mtf_analyzer import analyze_single_timeframe, TimeframeAnalysis

logger = logging.getLogger(__name__)


@dataclass
class ParallelAnalysisConfig:
    """Configuration for parallel analysis execution."""
    max_concurrent_symbols: int = 10
    max_concurrent_timeframes: int = 5
    max_concurrent_indicators: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: float = 30.0
    data_fetch_retries: int = 2
    timeout_per_symbol_seconds: float = 30.0


@dataclass
class ParallelAnalysisResult:
    """Result from parallel analysis of a single symbol."""
    symbol: str
    success: bool
    execution_time_ms: float
    
    # Component results
    indicators: Optional[IndicatorSummary] = None
    market_structure: Optional[MarketStructure] = None
    volume_analysis: Optional[VolumeAnalysis] = None
    timeframe_analyses: Dict[str, TimeframeAnalysis] = field(default_factory=dict)
    
    # Aggregated signals
    overall_signal: str = "NEUTRAL"
    confidence: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)


class ProfessionalParallelAnalyzer:
    """
    Professional-grade parallel analyzer for multi-symbol analysis.
    
    This class orchestrates concurrent execution of:
    - Data fetching across multiple symbols
    - Indicator calculations in parallel
    - Market structure analysis
    - Volume profiling
    - Multi-timeframe alignment
    
    All operations are thread-safe and optimized for high throughput.
    """
    
    def __init__(
        self,
        fetch_ohlcv: Callable[[str, str, int], pd.DataFrame],
        settings: Optional[Dict[str, Any]] = None,
        config: Optional[ParallelAnalysisConfig] = None,
    ):
        self.fetch_ohlcv = fetch_ohlcv
        self.settings = settings or {}
        self.config = config or ParallelAnalysisConfig()
        
        # Thread-safe storage
        self._data_cache: ThreadSafeDict[pd.DataFrame] = ThreadSafeDict()
        self._results_cache: ThreadSafeDict[ParallelAnalysisResult] = ThreadSafeDict()
        
        # Thread manager
        thread_config = ThreadPoolConfig(
            max_workers=max(
                self.config.max_concurrent_symbols,
                self.config.max_concurrent_timeframes,
            ),
            adaptive=True,
            enable_metrics=True,
        )
        self._manager = get_thread_manager(thread_config)
        
        # Data fetcher with rate limiting
        self._data_fetcher = ParallelDataFetcher(
            max_concurrent=self.config.max_concurrent_symbols,
            retry_attempts=self.config.data_fetch_retries,
            cache_ttl_seconds=self.config.cache_ttl_seconds,
        )
    
    def _fetch_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data with caching."""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        if self.config.enable_caching:
            cached = self._data_cache.get(cache_key)
            if cached is not None:
                return cached
        
        df = self.fetch_ohlcv(symbol, timeframe, limit)
        
        if self.config.enable_caching:
            self._data_cache.set(cache_key, df)
        
        return df
    
    def _analyze_indicators_parallel(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> IndicatorSummary:
        """Calculate all indicators (already optimized internally)."""
        return calculate_all_indicators(df, config)
    
    def _analyze_single_symbol(
        self,
        symbol: str,
        timeframes: List[str],
    ) -> ParallelAnalysisResult:
        """
        Analyze a single symbol with parallel component execution.
        
        This runs indicators, structure, and volume analysis concurrently.
        """
        start_time = time.perf_counter()
        errors: List[str] = []
        
        primary_tf = self.settings.get("strategy", {}).get("impulse_timeframe", "1h")
        
        try:
            # Fetch primary timeframe data
            df = self._fetch_data(symbol, primary_tf, 500)
            
            if len(df) < 50:
                return ParallelAnalysisResult(
                    symbol=symbol,
                    success=False,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    errors=["Insufficient data for analysis"],
                )
            
            # Run component analyses in parallel using thread manager
            analysis_tasks = {
                "indicators": lambda: self._analyze_indicators_parallel(
                    df, self.settings.get("deep_analysis", {})
                ),
                "structure": lambda: analyze_market_structure(
                    df, self.settings.get("market_structure", {})
                ),
                "volume": lambda: analyze_volume(
                    df, self.settings.get("volume_profile", {})
                ),
            }
            
            # Execute all analysis tasks concurrently
            task_results = self._manager.execute_concurrent(analysis_tasks)
            
            # Extract results
            indicators = None
            structure = None
            volume = None
            
            if task_results.get("indicators") and task_results["indicators"].success:
                indicators = task_results["indicators"].result
            else:
                errors.append(f"Indicator analysis failed: {task_results.get('indicators', {})}")
            
            if task_results.get("structure") and task_results["structure"].success:
                structure = task_results["structure"].result
            else:
                errors.append(f"Structure analysis failed")
            
            if task_results.get("volume") and task_results["volume"].success:
                volume = task_results["volume"].result
            else:
                errors.append(f"Volume analysis failed")
            
            # Multi-timeframe analysis in parallel
            tf_analyses: Dict[str, TimeframeAnalysis] = {}
            
            def analyze_tf(tf: str) -> Tuple[str, TimeframeAnalysis]:
                try:
                    tf_df = self._fetch_data(symbol, tf, 500)
                    return (tf, analyze_single_timeframe(tf_df, tf))
                except Exception as e:
                    return (tf, None)
            
            tf_results = self._manager.map_parallel(
                analyze_tf,
                timeframes,
                task_prefix=f"{symbol}_tf",
            )
            
            for result in tf_results:
                if result.success and result.result:
                    tf, analysis = result.result
                    if analysis:
                        tf_analyses[tf] = analysis
            
            # Calculate overall signal
            overall_signal, confidence = self._aggregate_signals(
                indicators, structure, volume, tf_analyses
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ParallelAnalysisResult(
                symbol=symbol,
                success=True,
                execution_time_ms=execution_time,
                indicators=indicators,
                market_structure=structure,
                volume_analysis=volume,
                timeframe_analyses=tf_analyses,
                overall_signal=overall_signal,
                confidence=confidence,
                errors=errors,
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Analysis failed for {symbol}: {e}")
            return ParallelAnalysisResult(
                symbol=symbol,
                success=False,
                execution_time_ms=execution_time,
                errors=[str(e)],
            )
    
    def _aggregate_signals(
        self,
        indicators: Optional[IndicatorSummary],
        structure: Optional[MarketStructure],
        volume: Optional[VolumeAnalysis],
        tf_analyses: Dict[str, TimeframeAnalysis],
    ) -> Tuple[str, float]:
        """Aggregate all component signals into overall signal."""
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        # Indicator contribution (weight: 2.5)
        if indicators:
            weight = 2.5
            total_weight += weight
            if indicators.overall_signal in ("STRONG_BUY", "BUY"):
                bullish_score += weight * indicators.overall_strength
            elif indicators.overall_signal in ("STRONG_SELL", "SELL"):
                bearish_score += weight * indicators.overall_strength
        
        # Structure contribution (weight: 3.0)
        if structure:
            weight = 3.0
            total_weight += weight
            if structure.trend == "BULLISH":
                bullish_score += weight * structure.trend_strength
            elif structure.trend == "BEARISH":
                bearish_score += weight * structure.trend_strength
        
        # Volume contribution (weight: 2.0)
        if volume:
            weight = 2.0
            total_weight += weight
            if volume.signal == "ACCUMULATION":
                bullish_score += weight * volume.strength
            elif volume.signal == "DISTRIBUTION":
                bearish_score += weight * volume.strength
        
        # MTF contribution (weight: 3.0)
        if tf_analyses:
            weight = 3.0
            total_weight += weight
            buy_count = sum(1 for tf in tf_analyses.values() if tf.signal == "BUY")
            sell_count = sum(1 for tf in tf_analyses.values() if tf.signal == "SELL")
            total_tf = len(tf_analyses)
            
            if total_tf > 0:
                if buy_count > sell_count:
                    bullish_score += weight * (buy_count / total_tf)
                elif sell_count > buy_count:
                    bearish_score += weight * (sell_count / total_tf)
        
        # Calculate overall
        if total_weight == 0:
            return "NEUTRAL", 50.0
        
        bullish_ratio = bullish_score / total_weight
        bearish_ratio = bearish_score / total_weight
        
        if bullish_ratio > 0.6:
            signal = "STRONG_BUY" if bullish_ratio > 0.75 else "BUY"
        elif bearish_ratio > 0.6:
            signal = "STRONG_SELL" if bearish_ratio > 0.75 else "SELL"
        else:
            signal = "NEUTRAL"
        
        confidence = max(bullish_ratio, bearish_ratio) * 100
        return signal, min(95.0, max(5.0, confidence))
    
    def analyze_symbols_parallel(
        self,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_result: Optional[Callable[[str, ParallelAnalysisResult], None]] = None,
    ) -> List[ParallelAnalysisResult]:
        """
        Analyze multiple symbols in parallel.
        
        This is the main entry point for parallel multi-symbol analysis.
        
        Args:
            symbols: List of symbols to analyze
            timeframes: Timeframes for MTF analysis (default: 1d, 4h, 1h, 15m, 5m)
            on_progress: Progress callback (completed, total, current_symbol)
            on_result: Called when each result is ready
        
        Returns:
            List of ParallelAnalysisResult objects
        """
        timeframes = timeframes or ["1d", "4h", "1h", "15m", "5m"]
        results: List[ParallelAnalysisResult] = []
        
        def analyze_symbol_wrapper(symbol: str) -> ParallelAnalysisResult:
            result = self._analyze_single_symbol(symbol, timeframes)
            if on_result:
                on_result(symbol, result)
            return result
        
        # Execute analysis in parallel
        task_results = self._manager.map_parallel(
            analyze_symbol_wrapper,
            symbols,
            task_prefix="symbol",
            on_progress=on_progress,
        )
        
        for result in task_results:
            if result.success and result.result:
                results.append(result.result)
            else:
                # Create error result
                results.append(ParallelAnalysisResult(
                    symbol=result.task_id.replace("symbol_", ""),
                    success=False,
                    execution_time_ms=result.execution_time_ms,
                    errors=[result.error or "Unknown error"],
                ))
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the thread manager."""
        metrics = self._manager.get_metrics()
        return {
            "total_tasks": metrics.total_tasks,
            "completed_tasks": metrics.completed_tasks,
            "failed_tasks": metrics.failed_tasks,
            "avg_execution_time_ms": metrics.avg_execution_time_ms,
            "active_threads": metrics.active_threads,
            "peak_threads": metrics.peak_threads,
        }


# =============================================================================
# Convenience Function for Drop-in Integration
# =============================================================================

def run_parallel_deep_analysis(
    symbols: List[str],
    fetch_ohlcv: Callable[[str, str, int], pd.DataFrame],
    settings: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[ParallelAnalysisResult]:
    """
    Run professional parallel deep analysis on multiple symbols.
    
    This is a convenience function that creates the analyzer and runs analysis.
    
    Args:
        symbols: List of symbols to analyze
        fetch_ohlcv: Function to fetch OHLCV data
        settings: Analysis settings
        on_progress: Optional progress callback
    
    Returns:
        List of analysis results
    
    Example:
        from src.core import binance_data as bd
        
        results = run_parallel_deep_analysis(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            fetch_ohlcv=bd.get_ohlcv,
            settings=settings,
            on_progress=lambda c, t, s: print(f"{c}/{t}: {s}")
        )
        
        for r in results:
            print(f"{r.symbol}: {r.overall_signal} ({r.confidence:.1f}%)")
    """
    analyzer = ProfessionalParallelAnalyzer(
        fetch_ohlcv=fetch_ohlcv,
        settings=settings,
    )
    
    return analyzer.analyze_symbols_parallel(
        symbols=symbols,
        on_progress=on_progress,
    )
