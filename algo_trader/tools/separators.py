import re

seplen = 120
dashes = "-" * seplen
hashes = "#" * seplen
stars = "*" * seplen
dbl_dash = "=" * seplen
cross = "+" * seplen
cashes = "$" * seplen
ATs = "@" * seplen
squiggles = "~" * seplen


def str_format(msg: str):
    """
    The str_format function removes leading whitespace from a string.
    It is used to remove the leading newline characters and any extra spaces that may be present in the docstring.

    :param msg:str: Pass a string to the function
    :return: A string with all leading whitespace removed from each line and common indentation removed
    :doc-author: Trelent
    """

    msg = re.sub(r"^\n+", "", msg)
    msg = re.sub(r"\n+$", "", msg)
    spaces = re.findall(r"^ +", msg, flags=re.MULTILINE)

    if len(spaces) > 0 and len(re.findall(r"^[^\s]", msg, flags=re.MULTILINE)) == 0:
        msg = re.sub(r"^%s" % (min(spaces)), "", msg, flags=re.MULTILINE).rstrip()

    return msg.strip()
