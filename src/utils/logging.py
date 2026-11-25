"""Loggers that print to the screen and, when asked, also to a log file."""
import logging
import os
import sys
from pathlib import Path


class _Screen(logging.StreamHandler):
    """Writes to the screen as it is when a message is logged (so a test or a caller can redirect it)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


class _LogFile(logging.Handler):
    """Appends each message to a log file and closes the file again at once, so no file is held open: on Windows an open file
    cannot be deleted or have its folder renamed. A message for a folder that no longer exists is skipped (never re-creates it)."""

    def __init__(self, path):
        super().__init__()
        self.baseFilename = os.path.abspath(path)

    def emit(self, record):
        if not os.path.isdir(os.path.dirname(self.baseFilename)):
            return
        try:
            with open(self.baseFilename, "a", encoding="utf-8", newline="\n") as f:
                f.write(self.format(record) + "\n")
        except Exception:      # a log line must never stop a run  # noqa: BLE001
            self.handleError(record)


def get_logger(name, log_file=None):
    """A logger that prints to the screen and, when log_file is given, also appends to that file (one file per logger: a new
    log_file replaces the earlier one, so a command never writes into the log of an earlier command)."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        screen = _Screen()
        screen.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(screen)
        log.propagate = False
    if log_file and not any(getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in log.handlers):
        for h in [h for h in log.handlers if getattr(h, "baseFilename", None)]:
            log.removeHandler(h)
            h.close()
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file = _LogFile(log_file)
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(file)
    return log
