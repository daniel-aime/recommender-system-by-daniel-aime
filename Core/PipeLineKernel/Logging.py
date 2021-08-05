import logging as log
import time
import functools


def log_timmer(func):
    """Logging time for executing any functions"""
    @functools.wraps(func)
    def timer(*args, **kwargs):
        start_time = time.perf_counter()
        vaue = func(*args, **kwargs)
        finish_time = time.perf_counter()
        run_time = finish_time - start_time
        print(f"Finished {func!r} in {run_time:.4f} seconds")
        return vaue
    return timer
