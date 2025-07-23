import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# Create a logs folder if it doesn't exist
log_folder = os.path.join(os.getcwd(), "logs")
os.makedirs(log_folder, exist_ok=True)

# Create full path to the log file
LOG_FILE_PATH = os.path.join(log_folder, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging setup complete.")
    logging.info(f"Log file created at: {LOG_FILE_PATH}")
    logging.info("This is an info message for testing the logger.")
    logging.error("This is an error message for testing the logger.")
    logging.warning("This is a warning message for testing the logger.")
    logging.debug("This is a debug message for testing the logger.")