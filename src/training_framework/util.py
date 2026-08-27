import importlib
import pickle
import pkgutil
import time
from functools import wraps
from abc import ABCMeta


def _is_serializable(self, obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False

def timestamp_str():
    ns_str = str(time.time_ns())

    # Convert the first part to time using float seconds
    base_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    # Grab the final 9 digits of the nanosecond string sequence
    fractional_ns = ns_str[-9:]

    return f"{base_time}_{fractional_ns}"

def context_entry(func):
    if getattr(func, "_context_entry_wrapper", None) is func:
        return func

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        outermost_call = not getattr(
            self,
            "_context_entry_in_progress",
            False,
        )
        if outermost_call:
            self._context_entry_in_progress = True

        try:
            result = func(self, *args, **kwargs)
            if outermost_call:
                self._active = True
            return result
        finally:
            if outermost_call:
                del self._context_entry_in_progress

    wrapper._context_entry_wrapper = wrapper
    return wrapper

def context_exit(func):
    if getattr(func, "_context_exit_wrapper", None) is func:
        return func

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        outermost_call = not getattr(
            self,
            "_context_exit_in_progress",
            False,
        )
        if outermost_call:
            self._context_exit_in_progress = True

        try:
            return func(self, *args, **kwargs)
        finally:
            if outermost_call:
                self._active = False
                del self._context_exit_in_progress

    wrapper._context_exit_wrapper = wrapper
    return wrapper

def requires_context(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_active") or not self._active:
            raise RuntimeError(f"This instance of {self.__class__.__name__} is not initialized yet!")
        return func(self, *args, **kwargs)
    return wrapper


class CaptureInitMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        init = cls.__init__

        # Avoid wrapping twice if a parent already wrapped it.
        if getattr(init, "_captures_init_args", False):
            return cls

        @wraps(init)
        def wrapped_init(self, *args, **kwargs):
            outermost_call = not getattr(
                self,
                "_capture_init_args_active",
                False,
            )
            if outermost_call:
                self._init_args = {
                    "args": args,
                    "kwargs": kwargs,
                }
                self._capture_init_args_active = True

            try:
                return init(self, *args, **kwargs)
            finally:
                if outermost_call:
                    del self._capture_init_args_active

        wrapped_init._captures_init_args = True
        cls.__init__ = wrapped_init
        return cls


def import_all_modules(package_name: str) -> None:
    package = importlib.import_module(package_name)

    if not hasattr(package, "__path__"):
        return

    prefix = package.__name__ + "."

    for module_info in pkgutil.walk_packages(
        package.__path__,
        prefix=prefix,
    ):
        importlib.import_module(module_info.name)


def format_execution_time(nanos: int) -> str:
    if nanos < 1_000:
        return f"{nanos} ns"
    elif nanos < 1_000_000:
        return f"{nanos / 1_000:.2f} µs"
    elif nanos < 1_000_000_000:
        return f"{nanos / 1_000_000:.2f} ms"
    else:
        seconds = nanos / 1_000_000_000
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{int(h)}h {int(m)}m {s:.2f}s"
        return f"{int(m)}m {s:.2f}s"
