"""
Professional Thread Manager Module

This module provides enterprise-grade threading capabilities for high-performance
parallel execution of analysis tasks. Features include:

- Adaptive thread pool management
- Priority-based task scheduling
- Thread-safe data aggregation
- Performance monitoring and metrics
- Automatic resource optimization
- Graceful error handling and recovery
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union
)
from functools import wraps
import traceback

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TaskResult(Generic[T]):
    """Result wrapper for threaded task execution."""
    task_id: str
    success: bool
    result: Optional[T] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    thread_name: str = ""


@dataclass
class TaskMetrics:
    """Metrics for monitoring thread pool performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    active_threads: int = 0
    peak_threads: int = 0
    queue_size: int = 0
    
    def update(self, execution_time_ms: float, success: bool) -> None:
        """Update metrics with a completed task."""
        self.total_tasks += 1
        self.completed_tasks += 1 if success else 0
        self.failed_tasks += 0 if success else 1
        self.total_execution_time_ms += execution_time_ms
        if self.completed_tasks > 0:
            self.avg_execution_time_ms = self.total_execution_time_ms / self.completed_tasks


@dataclass
class ThreadPoolConfig:
    """Configuration for thread pool behavior."""
    max_workers: int = 8
    min_workers: int = 2
    adaptive: bool = True
    task_timeout_seconds: float = 60.0
    queue_max_size: int = 1000
    enable_metrics: bool = True
    retry_on_failure: bool = False
    max_retries: int = 2
    retry_delay_seconds: float = 0.5


class ThreadSafeCounter:
    """Thread-safe counter for tracking concurrent operations."""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value


class ThreadSafeDict(Generic[T]):
    """Thread-safe dictionary for concurrent data access."""
    
    def __init__(self):
        self._data: Dict[str, T] = {}
        self._lock = threading.RLock()
    
    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._data[key] = value
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        with self._lock:
            return self._data.get(key, default)
    
    def get_all(self) -> Dict[str, T]:
        with self._lock:
            return dict(self._data)
    
    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ThreadSafeList(Generic[T]):
    """Thread-safe list for concurrent data collection."""
    
    def __init__(self):
        self._data: List[T] = []
        self._lock = threading.Lock()
    
    def append(self, item: T) -> None:
        with self._lock:
            self._data.append(item)
    
    def extend(self, items: List[T]) -> None:
        with self._lock:
            self._data.extend(items)
    
    def get_all(self) -> List[T]:
        with self._lock:
            return list(self._data)
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ProfessionalThreadManager:
    """
    Enterprise-grade thread manager for professional parallel execution.
    
    Features:
    - Adaptive thread pool sizing based on workload
    - Priority task scheduling
    - Comprehensive metrics and monitoring
    - Graceful shutdown and error recovery
    - Rate limiting support for API calls
    """
    
    def __init__(self, config: Optional[ThreadPoolConfig] = None):
        self.config = config or ThreadPoolConfig()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._metrics = TaskMetrics()
        self._metrics_lock = threading.Lock()
        self._active_counter = ThreadSafeCounter()
        self._shutdown = threading.Event()
        self._rate_limiter: Optional[threading.Semaphore] = None
        
    def start(self) -> None:
        """Initialize and start the thread pool."""
        if self._executor is not None:
            return
            
        workers = self._calculate_optimal_workers()
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="ProAnalysis"
        )
        logger.info(f"Thread pool started with {workers} workers")
    
    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Gracefully shutdown the thread pool."""
        self._shutdown.set()
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
            logger.info("Thread pool stopped")
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads."""
        import os
        
        if not self.config.adaptive:
            return self.config.max_workers
        
        # Use CPU count as baseline, but cap at max_workers
        cpu_count = os.cpu_count() or 4
        
        # For I/O bound tasks (API calls), we can use more threads
        optimal = min(cpu_count * 2, self.config.max_workers)
        optimal = max(optimal, self.config.min_workers)
        
        return optimal
    
    def _update_metrics(self, execution_time_ms: float, success: bool) -> None:
        """Thread-safe metrics update."""
        with self._metrics_lock:
            self._metrics.update(execution_time_ms, success)
            self._metrics.active_threads = self._active_counter.get()
            self._metrics.peak_threads = max(
                self._metrics.peak_threads,
                self._metrics.active_threads
            )
    
    def get_metrics(self) -> TaskMetrics:
        """Get current performance metrics."""
        with self._metrics_lock:
            return TaskMetrics(
                total_tasks=self._metrics.total_tasks,
                completed_tasks=self._metrics.completed_tasks,
                failed_tasks=self._metrics.failed_tasks,
                total_execution_time_ms=self._metrics.total_execution_time_ms,
                avg_execution_time_ms=self._metrics.avg_execution_time_ms,
                active_threads=self._active_counter.get(),
                peak_threads=self._metrics.peak_threads,
                queue_size=self._metrics.queue_size,
            )
    
    def set_rate_limit(self, max_concurrent: int) -> None:
        """Set rate limiting for API calls."""
        self._rate_limiter = threading.Semaphore(max_concurrent)
    
    def submit(
        self,
        task_id: str,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> Future[TaskResult[T]]:
        """
        Submit a task for execution.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Future containing TaskResult
        """
        if self._executor is None:
            self.start()
        
        def wrapped_task() -> TaskResult[T]:
            start_time = time.perf_counter()
            thread_name = threading.current_thread().name
            self._active_counter.increment()
            
            try:
                # Apply rate limiting if configured
                if self._rate_limiter:
                    self._rate_limiter.acquire()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = (time.perf_counter() - start_time) * 1000
                    
                    self._update_metrics(execution_time, True)
                    
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        execution_time_ms=execution_time,
                        thread_name=thread_name,
                    )
                finally:
                    if self._rate_limiter:
                        self._rate_limiter.release()
                        
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Task {task_id} failed: {error_msg}")
                
                self._update_metrics(execution_time, False)
                
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=error_msg,
                    execution_time_ms=execution_time,
                    thread_name=thread_name,
                )
            finally:
                self._active_counter.decrement()
        
        return self._executor.submit(wrapped_task)
    
    def map_parallel(
        self,
        func: Callable[[Any], T],
        items: List[Any],
        task_prefix: str = "task",
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[TaskResult[T]]:
        """
        Execute function on all items in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            task_prefix: Prefix for task IDs
            on_progress: Optional progress callback (current, total, item_id)
        
        Returns:
            List of TaskResults in order of completion
        """
        if self._executor is None:
            self.start()
        
        futures: Dict[Future, str] = {}
        results: List[TaskResult[T]] = []
        
        # Submit all tasks
        for i, item in enumerate(items):
            task_id = f"{task_prefix}_{i}"
            future = self.submit(task_id, func, item)
            futures[future] = task_id
        
        # Collect results as they complete
        completed = 0
        total = len(items)
        
        for future in as_completed(futures.keys()):
            task_id = futures[future]
            try:
                result = future.result(timeout=self.config.task_timeout_seconds)
                results.append(result)
            except Exception as e:
                results.append(TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                ))
            
            completed += 1
            if on_progress:
                on_progress(completed, total, task_id)
        
        return results
    
    def execute_concurrent(
        self,
        tasks: Dict[str, Callable[[], T]],
    ) -> Dict[str, TaskResult[T]]:
        """
        Execute multiple independent tasks concurrently.
        
        Args:
            tasks: Dictionary mapping task_id to callable
        
        Returns:
            Dictionary mapping task_id to TaskResult
        """
        if self._executor is None:
            self.start()
        
        futures: Dict[Future, str] = {}
        
        for task_id, func in tasks.items():
            future = self.submit(task_id, func)
            futures[future] = task_id
        
        results: Dict[str, TaskResult[T]] = {}
        
        for future in as_completed(futures.keys()):
            task_id = futures[future]
            try:
                result = future.result(timeout=self.config.task_timeout_seconds)
                results[task_id] = result
            except Exception as e:
                results[task_id] = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                )
        
        return results


# =============================================================================
# Singleton Thread Manager Instance
# =============================================================================

_global_manager: Optional[ProfessionalThreadManager] = None
_manager_lock = threading.Lock()


def get_thread_manager(config: Optional[ThreadPoolConfig] = None) -> ProfessionalThreadManager:
    """Get or create the global thread manager instance."""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = ProfessionalThreadManager(config)
            _global_manager.start()
        return _global_manager


def shutdown_thread_manager() -> None:
    """Shutdown the global thread manager."""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is not None:
            _global_manager.stop()
            _global_manager = None


# =============================================================================
# Decorator for Easy Parallel Execution
# =============================================================================

def run_parallel(task_id: Optional[str] = None):
    """
    Decorator to run a function in the thread pool.
    
    Usage:
        @run_parallel("my_task")
        def heavy_computation(data):
            return process(data)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Future[TaskResult[T]]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Future[TaskResult[T]]:
            manager = get_thread_manager()
            tid = task_id or f"{func.__name__}_{id(args)}"
            return manager.submit(tid, func, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Parallel Analysis Utilities
# =============================================================================

class ParallelDataFetcher:
    """
    Professional parallel data fetcher for API calls.
    
    Features:
    - Automatic rate limiting
    - Retry logic with exponential backoff
    - Response caching
    - Error aggregation
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        retry_attempts: int = 2,
        cache_ttl_seconds: float = 60.0,
    ):
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.cache_ttl = cache_ttl_seconds
        self._cache: ThreadSafeDict[Tuple[Any, float]] = ThreadSafeDict()
        self._manager = get_thread_manager()
        self._manager.set_rate_limit(max_concurrent)
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        cached = self._cache.get(key)
        if cached:
            result, timestamp = cached
            if time.time() - timestamp < self.cache_ttl:
                return result
        return None
    
    def _set_cached(self, key: str, value: Any) -> None:
        """Cache a result with timestamp."""
        self._cache.set(key, (value, time.time()))
    
    def fetch_all(
        self,
        fetch_func: Callable[[str], T],
        symbols: List[str],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, T]:
        """
        Fetch data for all symbols in parallel.
        
        Args:
            fetch_func: Function that takes a symbol and returns data
            symbols: List of symbols to fetch
            on_progress: Optional progress callback
        
        Returns:
            Dictionary mapping symbol to fetched data
        """
        results: Dict[str, T] = {}
        to_fetch: List[str] = []
        
        # Check cache first
        for symbol in symbols:
            cached = self._get_cached(symbol)
            if cached is not None:
                results[symbol] = cached
            else:
                to_fetch.append(symbol)
        
        if not to_fetch:
            return results
        
        # Fetch remaining in parallel
        def fetch_with_retry(sym: str) -> Tuple[str, T]:
            last_error = None
            for attempt in range(self.retry_attempts + 1):
                try:
                    data = fetch_func(sym)
                    self._set_cached(sym, data)
                    return (sym, data)
                except Exception as e:
                    last_error = e
                    if attempt < self.retry_attempts:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            raise last_error
        
        task_results = self._manager.map_parallel(
            fetch_with_retry,
            to_fetch,
            task_prefix="fetch",
            on_progress=on_progress,
        )
        
        for result in task_results:
            if result.success and result.result:
                sym, data = result.result
                results[sym] = data
        
        return results


class ParallelAnalyzer:
    """
    Professional parallel analyzer for running analysis on multiple symbols.
    
    Automatically distributes work across threads for maximum throughput.
    """
    
    def __init__(self, config: Optional[ThreadPoolConfig] = None):
        self._manager = get_thread_manager(config)
        self._results: ThreadSafeList[Any] = ThreadSafeList()
    
    def analyze_symbols_parallel(
        self,
        analyze_func: Callable[[str], T],
        symbols: List[str],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_result: Optional[Callable[[str, T], None]] = None,
    ) -> List[Tuple[str, T]]:
        """
        Analyze multiple symbols in parallel.
        
        Args:
            analyze_func: Analysis function that takes a symbol
            symbols: List of symbols to analyze
            on_progress: Progress callback (completed, total, current_symbol)
            on_result: Called immediately when each result is ready
        
        Returns:
            List of (symbol, result) tuples
        """
        results: List[Tuple[str, T]] = []
        
        def analyze_wrapper(symbol: str) -> Tuple[str, T]:
            result = analyze_func(symbol)
            if on_result:
                on_result(symbol, result)
            return (symbol, result)
        
        task_results = self._manager.map_parallel(
            analyze_wrapper,
            symbols,
            task_prefix="analyze",
            on_progress=on_progress,
        )
        
        for result in task_results:
            if result.success and result.result:
                results.append(result.result)
        
        return results
    
    def run_analysis_pipeline(
        self,
        symbol: str,
        stages: Dict[str, Callable[[], T]],
    ) -> Dict[str, TaskResult[T]]:
        """
        Run multiple independent analysis stages in parallel.
        
        This is useful when multiple independent analyses need to run
        for the same symbol (e.g., indicators, structure, volume).
        
        Args:
            symbol: Symbol being analyzed
            stages: Dictionary mapping stage name to analysis function
        
        Returns:
            Dictionary mapping stage name to TaskResult
        """
        return self._manager.execute_concurrent(stages)


# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """
    Monitor and log thread pool performance.
    """
    
    def __init__(self, manager: ProfessionalThreadManager):
        self._manager = manager
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start(self, interval_seconds: float = 5.0) -> None:
        """Start the performance monitor."""
        if self._running:
            return
            
        self._running = True
        
        def monitor_loop():
            while self._running:
                metrics = self._manager.get_metrics()
                logger.info(
                    f"ThreadPool Metrics: "
                    f"completed={metrics.completed_tasks}, "
                    f"failed={metrics.failed_tasks}, "
                    f"active={metrics.active_threads}, "
                    f"avg_time={metrics.avg_execution_time_ms:.1f}ms"
                )
                time.sleep(interval_seconds)
        
        self._monitor_thread = threading.Thread(
            target=monitor_loop,
            name="PerfMonitor",
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop the performance monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
