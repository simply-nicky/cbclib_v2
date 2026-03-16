import threading
from threading import local

class CPUConfig:
    def __init__(self, num_threads: int = 1):
        self.num_threads = num_threads
        self._old : int | None = None

    @staticmethod
    def _validate_num_threads(n: int) -> int:
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"num_threads must be an integer >= 1, got {n}")
        return n

    def __setattr__(self, name, value):
        if name == "num_threads":
            value = self._validate_num_threads(value)
        super().__setattr__(name, value)

    def __enter__(self):
        self._old = get_cpu_config().num_threads
        get_cpu_config().num_threads = self.num_threads
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old is not None:
            get_cpu_config().num_threads = self._old
            self._old = None

    def effective_num_threads(self) -> int:
        # Force 1 thread when running in a non-main thread to avoid nested parallelism.
        if threading.current_thread() is not threading.main_thread():
            return 1
        return self.num_threads

_default_num_threads = 1
_thread_local = local()

def get_cpu_config() -> CPUConfig:
    if not hasattr(_thread_local, "config"):
        _thread_local.config = CPUConfig(_default_num_threads)
    return getattr(_thread_local, "config",)

def set_cpu_config(num_threads: int) -> None:
    config = get_cpu_config()
    config.num_threads = num_threads

def reset_cpu_config() -> None:
    if hasattr(_thread_local, "config"):
        delattr(_thread_local, "config")
