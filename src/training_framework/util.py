import functools
import pickle
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
    @functools.wraps(func)
    def wrapper(self):
        res = func(self)
        self._active = True
        return res
    return wrapper

def context_exit(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._active = False
        return func(self, *args, **kwargs)
    return wrapper

def requires_context(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_active") or not self._active:
            raise RuntimeError("Use within 'with'!")
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
            # Store the exact call shape for later reconstruction.
            self._init_args = {
                "args": args,
                "kwargs": kwargs,
            }

            return init(self, *args, **kwargs)

        wrapped_init._captures_init_args = True
        cls.__init__ = wrapped_init
        return cls