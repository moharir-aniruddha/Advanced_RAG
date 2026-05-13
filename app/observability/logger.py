import logging

import os


LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


logging.basicConfig(
    filename=f"{LOG_DIR}/rag_logs.log",
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)


logger = logging.getLogger(__name__)