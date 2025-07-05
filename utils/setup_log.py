# utils/setup_log.py

from pathlib import Path
import logging
from datetime import datetime, timezone

LOG_DIR = Path(__file__).parent.parent / ".cache" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_app_logger():
    logger = logging.getLogger("hubspot-assistant")
    logger.setLevel(logging.INFO)
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOG_DIR / f"app_{ts}.log", encoding="utf-8")
    fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return logger

# Initialize the logger immediately when this module is imported
logger = setup_app_logger()
