# utils/setup_log.py

from pathlib import Path
import logging
from datetime import datetime, timezone

LOG_DIR = Path(__file__).parent.parent / ".cache" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_app_logger():
    # Create (or get) a logger named "myapp"
    logger = logging.getLogger("myapp")
    # Lower this to DEBUG so that we can see debug messages
    logger.setLevel(logging.DEBUG)

    # If no FileHandler exists yet, add one
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(LOG_DIR / f"app_{ts}.log", encoding="utf-8")
        fmt = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Also add a StreamHandler (console) if none exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)  # Let console show DEBUG+ messages
        fmt = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Prevent messages from being “lost” if other modules use root logger
    logger.propagate = False

    return logger

# Initialize the logger immediately when this module is imported
logger = setup_app_logger()
