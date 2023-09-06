import datetime


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%y-%m-%d_%H-%M-%S")

    return cur