import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail)

    def error_message_detail(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        return f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{error_message}]"

    def __str__(self):
        return self.error_message

