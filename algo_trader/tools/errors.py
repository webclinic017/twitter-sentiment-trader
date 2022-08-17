import traceback

from .separators import hashes, str_format


def err_catch(exc, traceback_msg=True):
    """
    The err_catch function is a decorator that catches exceptions and formats
    them to be used in loggers.

    :param exc: Get the exception that is being handled
    :param traceback_msg: Determine if the traceback should be printed
    :return: The error message and the traceback
    :doc-author: Trelent
    """
    msg = ""
    if "msg" in dir(exc):
        msg += exc.msg
    if traceback_msg:
        frm = traceback.format_exc().splitlines()
        stack = traceback.format_stack()[:-1]
        msg += f"\n{frm[0]}\n"
        msg += "".join(stack)
        msg += "\n".join(frm[1:])
    return msg


class ThreaderFailed(SystemError):
    def __init__(self, tag, fcnt):
        self.msg = str_format(
            f"""
        {hashes}
        ({tag}): {fcnt} thread(s) failed! Aborting load!
        {hashes}
        """
        )
        super().__init__()
