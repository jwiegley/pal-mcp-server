"""Server logging resources must survive reconfiguration without leaking."""

import importlib
import logging
from logging.handlers import RotatingFileHandler


def test_reload_closes_old_log_files_and_replaces_activity_handler():
    import server

    loggers = (logging.getLogger(), logging.getLogger("mcp_activity"))
    previous = [
        handler for logger in loggers for handler in logger.handlers if isinstance(handler, RotatingFileHandler)
    ]
    assert len(previous) == 2
    streams = [handler.stream for handler in previous]
    assert all(stream is not None and not stream.closed for stream in streams)

    importlib.reload(server)

    assert all(stream.closed for stream in streams)
    for logger in loggers:
        current = [handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)]
        assert len(current) == 1
        assert current[0] not in previous
        assert not current[0].stream.closed
