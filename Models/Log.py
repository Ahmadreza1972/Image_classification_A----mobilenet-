import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir, log_filename=None):
        os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

        # Set default filename with date if not provided
        if log_filename is None:
            log_filename = f"training_{datetime.now().strftime('%Y-%m-%d')}.log"

        log_file = os.path.join(log_dir, log_filename)
        open(log_file, 'w').close()
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            filemode="a",  # Append logs instead of overwriting
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

        # Also log to console
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.console_handler.setFormatter(formatter)

        # Get the logger and attach handler
        self.logger = logging.getLogger()
        self.logger.addHandler(self.console_handler)

    def log(self, message, level="info"):
        """Logs messages at different levels: info, warning, error."""
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)  # Default to info if level is unknown
