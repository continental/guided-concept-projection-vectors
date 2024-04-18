'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

import logging
import time
from typing import Type
import os
import traceback
import sys


def _get_traceback_msg(stack_idx: int = -3
                       ) -> str:
    """
    Gets traceback stack and converts element 'stack_idx' into the message
    Message: filename.function:line_number

    Kwrgs:
        stack_idx: element of traceback stack to convert into message

    Returns:
        message: str
    """
    tb = traceback.extract_stack()
    frame = tb[stack_idx]

    module = os.path.basename(frame.filename)
    func = frame.name
    lineno = frame.lineno

    return f"{module}.{func}:{lineno}"


def init_logger(file_name: str = None,
                log_level: int = logging.INFO
                ) -> None:
    """
    Init logger. Time is converted to GMT

    Kwargs:
        file_name: log to file (FileHandler), log to console if None (StreamHandler)
        log_level: level of logging
    """
    f = "%(asctime)s.%(msecs)d %(levelname)s %(message)s"
    configs = {'format': f,
               'datefmt': '%Y-%m-%d %H:%M:%S',
               'level': log_level}

    if file_name is not None:
        configs['filename'] = file_name    

    logging.basicConfig(**configs)

    logging.Formatter.converter = time.gmtime


def log_assert(assertion: bool,
               msg: str
               ) -> None:
    """
    Assert and log assertion.
    Exit, if assertion is wrong.

    Args:
        assertion: assertion expression (or boolean result)
        msg: assertion message if False
    """
    try:
        assert assertion
        log_msg("Succesfull assertion", logging.DEBUG, -4)
    except AssertionError as e:
        tb_msg = _get_traceback_msg()
        err_msg = ' '.join([tb_msg, "Assertion Error:", msg])
        logging.error(err_msg)
        sys.exit(1)


def log_error(error_cls: Type[Exception],
              msg: str
              ) -> None:
    """
    Raise error, log it and exit.

    Args:
        error_cls: Error class
        msg: error message
    """
    try:
        raise error_cls(msg)
    except error_cls:
        tb_msg = _get_traceback_msg()
        err_msg = ' '.join([tb_msg, msg])
        logging.error(err_msg)
        sys.exit(1)
    

def log_msg(msg: str,
            log_lvl: int,
            stack_idx: int = -3
            ) -> None:
    """
    Log message.
    Message: traceback_str[filename.function:line_number] message_text

    Args:
        msg: message text
        log_lvl: log level

    Kwargs
        stack_idx: index in traceback stack to track the source of message, defaults to -3: source_function() -> log_msg() -> _get_traceback_msg() -> traceback.extract_stack()
    """
    tb_msg = _get_traceback_msg(stack_idx)
    log_msg = ' '.join([tb_msg, msg])
    logging.log(log_lvl, log_msg)


def log_debug(msg: str
              ) -> None:
    """
    Log DEBUG message.

    Args:
        msg: message text
    """
    log_msg(msg, logging.DEBUG, -4)


def log_info(msg: str
              ) -> None:
    """
    Log INFO message.

    Args:
        msg: message text
    """
    log_msg(msg, logging.INFO, -4)
