import logging
import os

def configure_logging(log_folder: str, log_filename: str):
    try:
        logger = logging.getLogger(log_filename)

        if logger.hasHandlers():
            logger.handlers.clear()

        current_directory = os.path.dirname(os.path.realpath(__file__))
        log_folder_path = os.path.join(current_directory, log_folder)
        os.makedirs(log_folder_path, exist_ok=True)
        log_filepath = os.path.join(log_folder_path, log_filename)

        # Set logging level
        logger.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        # Optionally, add a stream handler to see logs on the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger
    except Exception as e:
        raise RuntimeError(f"Error while configuring logging: {e}")
