"""System Logger Setup - Handles automatic file logging and crash interception."""
import os
import sys
import logging
import threading
import traceback
from logging.handlers import RotatingFileHandler
import config

def setup_system_logger():
    """Configures rotating file logging and global exception hooks."""
    
    # 1. Ensure the LOGS directory exists inside your database folder
    log_dir = os.path.join(config.DB_PATH, "LOGS")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "system_activity.log")
    error_file_path = os.path.join(log_dir, "system_errors.log")

    # 2. Create a Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if setup is called multiple times
    if root_logger.handlers:
        return root_logger

    # 3. Formatting template (Timestamp | Log Level | Thread/Module | Message)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. General Activity Handler (Rotates after 5 MB, keeps last 5 backup archives)
    activity_handler = RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    activity_handler.setLevel(logging.INFO)
    activity_handler.setFormatter(formatter)
    root_logger.addHandler(activity_handler)

    # 5. Dedicated Error Handler (Only saves WARNING, ERROR, and CRITICAL to a separate file)
    error_handler = RotatingFileHandler(
        error_file_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # 6. Console Handler (So you can still see logs in CMD when testing!)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --- 7. GLOBAL CRASH INTERCEPTORS ---
    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        """Catches any fatal crash in the main GUI thread."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical("🚨 FATAL UNHANDLED GUI CRASH:", exc_info=(exc_type, exc_value, exc_traceback))

    def handle_thread_exception(args):
        """Catches crashes inside RTSPGrabber, VideoRecorder, or AI workers!"""
        root_logger.critical(
            f"🚨 FATAL BACKGROUND THREAD CRASH in [{args.thread.name}]:", 
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )

    # Override Windows/Python default error hooks
    sys.excepthook = handle_unhandled_exception
    threading.excepthook = handle_thread_exception

    logging.info("==================================================")
    logging.info("🏥 HOSPITAL AI MASTER LOGGING ENGINE INITIALIZED")
    logging.info(f"📁 Logs saving to: {log_dir}")
    logging.info("==================================================")

    return root_logger