import socket


def get_server_unused_port(port_list: list) -> int:
    """
    Returns a port number from a pre-defined list of target ports. Used to start a server of a free port.
    """
    for idx, port in enumerate(port_list):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) == 0:
                print(f"Port {port} in use. Trying next available...")
            else:
                return port
