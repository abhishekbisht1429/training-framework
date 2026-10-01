"""What a worker sends its parent when it fails.

Built in one place for both ways a worker reports -- the session's own exit
and the worker loop around it -- so the parent reads one format.
"""

import os
import traceback
from typing import Any

FAILURE_REPORT_TYPE = "error"


def exception_type_name(error_type: type) -> str:
    """`module.QualName` of an exception type; builtins by name alone."""
    module = error_type.__module__
    name = error_type.__qualname__
    return name if module == "builtins" else f"{module}.{name}"


def worker_failure_report(rank: int, error: BaseException) -> dict[str, Any]:
    """The report a worker of rank `rank` sends when `error` ends it.

    The traceback is formatted from `error` itself, so the report does not
    depend on being built inside the `except` that caught it.
    """
    return {
        "type": FAILURE_REPORT_TYPE,
        "rank": rank,
        "pid": os.getpid(),
        "exception_type": exception_type_name(type(error)),
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }
