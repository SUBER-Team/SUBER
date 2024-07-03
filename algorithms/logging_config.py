# logging_config.py
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    # Check if the root logger already has handlers (to prevent adding multiple handlers)
    if not logging.getLogger().hasHandlers():
        # Set up basic configuration for the root logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create a rotating file handler
        handler = RotatingFileHandler('suber.log', maxBytes=2000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the rotating file handler to the root logger
        logging.getLogger().addHandler(handler)

# Call setup_logging to ensure the configuration is applied
setup_logging()

