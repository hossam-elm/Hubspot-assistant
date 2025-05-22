
from pathlib import Path
import logging
from datetime import datetime, timezone

LOG_DIR = Path(__file__).parent.parent / ".cache" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_app_logger():
    logger = logging.getLogger("myapp")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(LOG_DIR / f"app_{ts}.log", encoding="utf-8")
        fmt = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = setup_app_logger()
