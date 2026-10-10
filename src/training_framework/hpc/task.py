"""`python -m training_framework.hpc.task`: the remote task entry point.

Its name is part of the execution protocol (rendered job scripts call it),
so it stays here; the implementation lives in `remote/task.py`.
"""

import sys

from training_framework.hpc.remote.task import main

if __name__ == "__main__":
    sys.exit(main())
