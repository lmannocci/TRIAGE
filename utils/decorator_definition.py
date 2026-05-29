
from functools import wraps
import time
from contextlib import contextmanager


def log_method(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        method_name = func.__name__
        self.lm.printl(f"{method_name} - START")
        result = func(self, *args, **kwargs)
        self.lm.printl(f"{method_name} - COMPLETED")
        return result
    return wrapper


@contextmanager
def measure_time(lm, when_print='always', i=None, K=None, label="Execution"):
    start_time = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start_time

    if when_print == 'always':
        lm.printl(f"{label}: {elapsed_time:.6f} seconds")
    elif when_print == 'every_K':
        lm.printK(i, K, f"{label}: {elapsed_time:.6f} seconds")
