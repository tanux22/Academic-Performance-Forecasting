import sys
import logging
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    """
    Generates a formatted error message.

    Args:
        error (Exception): The exception that occurred.
        error_detail (sys): The sys module for accessing exception details.

    Returns:
        str: A formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message= f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self):
        return self.error_message
    
    
if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys) 