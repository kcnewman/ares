import logging
import os
import sys

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("ares")
