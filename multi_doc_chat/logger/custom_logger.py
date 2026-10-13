
import os
import logging
from datetime import datetime
import structlog


class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name="multi_doc_chat"):
        logger_name = os.path.basename(name)

        # Configure root logger only once
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        if not root_logger.handlers:
            # File handler
            file_handler = logging.FileHandler(
                self.log_file_path,
                encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(message)s")
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter("%(message)s")
            )

            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(
                    fmt="iso",
                    utc=True,
                    key="timestamp"
                ),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(
                    to="event"
                ),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)


# Global logger instance available across the project
GLOBAL_LOGGER = CustomLogger().get_logger()
