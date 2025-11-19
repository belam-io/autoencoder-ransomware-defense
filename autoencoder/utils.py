import logging
import graypy

def get_logger_helper():
    logger = logging.getLogger("AnomalyDetector")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = graypy.GELFHTTPHandler(
            host="marc-module-car-subscribers.trycloudflare.com",
            port=443,  
            path="/gelf"
        )
        logger.addHandler(handler)

    logger.propagate = False
    return logger
