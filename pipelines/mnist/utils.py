import socket


def get_ip_address():
    """
    Fetches a client's external IP address
    :return: External IP address (str)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]