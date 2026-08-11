"""System-wide Logging Configuration with Bounded Rotation and Categories."""

import os
import logging
from logging.handlers import RotatingFileHandler
import config

class LogCategory:
    """Standardized tags for system-wide grep/parsing."""
    CAMERA = "[CAMERA]"
    AI = "[AI]"
    SESSION = "[SESSION]"
    RECORDING = "[RECORDING]"
    DATABASE = "[DATABASE]"
    NETWORK = "[NETWORK]"
    WATCHDOG = "[WATCHDOG]"
    HEALTH = "[HEALTH]"

def setup_logger():
    """Initializes the root logger with RotatingFileHandler and Console output."""
    log_dir = os.path.join(config.DB_PATH, "LOGS")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "system_events.log")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to prevent duplicate lines if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Clean format: 2026-08-10 14:22:11 | INFO    | [CAMERA][SINK_1] connected
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Bounded File Handler (Max 5MB per file, keep the last 5 backups)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=5 * 1024 * 1024, 
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler (For live terminal viewing)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info(f"{LogCategory.HEALTH} Structured logging initialized.")