import logging
import graypy

def get_logger_helper():
    logger = logging.getLogger('AnomalyDetector')
    logger.setLevel(logging.INFO)

    if not logger.handlers:     #
        handler = graypy.GELFUDPHandler('127.0.0.1', 12201)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
