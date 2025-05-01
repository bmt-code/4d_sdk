import time
import logging
from typing import Dict, Optional, Set


class CustomLogger:
    def __init__(self, name: str):
        self.name = name
        self._throttle_times: Dict[str, float] = {}  # message -> last time logged
        self._logged_once: Set[str] = set()  # messages that have been logged once

        # Configure the logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        # Create console handler with formatting
        if not self._logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(name)s] [%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def _should_log(
        self, message: str, throttle_sec: Optional[float], log_once: bool
    ) -> bool:
        """Determine if a message should be logged based on throttling and log-once rules."""
        if log_once:
            if message in self._logged_once:
                return False
            self._logged_once.add(message)

        if throttle_sec is not None:
            current_time = time.time()
            last_time = self._throttle_times.get(message, 0)
            if current_time - last_time < throttle_sec:
                return False
            self._throttle_times[message] = current_time

        return True

    def info(
        self, message: str, throttle_sec: Optional[float] = None, log_once: bool = False
    ):
        """
        Log an info message with optional throttling and log-once capabilities.

        Args:
            message: The message to log
            throttle_sec: If set, only log this message once every N seconds
            log_once: If True, only log this message once ever
        """
        if self._should_log(message, throttle_sec, log_once):
            self._logger.info(message)

    def warn(
        self, message: str, throttle_sec: Optional[float] = None, log_once: bool = False
    ):
        """Log a warning message with optional throttling and log-once capabilities."""
        if self._should_log(message, throttle_sec, log_once):
            self._logger.warning(message)

    def error(
        self, message: str, throttle_sec: Optional[float] = None, log_once: bool = False
    ):
        """Log an error message with optional throttling and log-once capabilities."""
        if self._should_log(message, throttle_sec, log_once):
            self._logger.error(message)

    def debug(
        self, message: str, throttle_sec: Optional[float] = None, log_once: bool = False
    ):
        """Log a debug message with optional throttling and log-once capabilities."""
        if self._should_log(message, throttle_sec, log_once):
            self._logger.debug(message)
