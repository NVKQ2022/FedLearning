"""
Unit tests for gRPC runner utilities and port management.
"""

import socket
from src.federated.grpc_runner import is_port_in_use, find_available_port


def test_is_port_in_use_and_find_available_port():
    # Bind a temporary socket to check detection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    bound_port = sock.getsockname()[1]

    try:
        assert is_port_in_use(bound_port) is True
        # find_available_port starting at bound_port should return a different free port
        free_port = find_available_port(start_port=bound_port)
        assert free_port != bound_port
        assert is_port_in_use(free_port) is False
    finally:
        sock.close()
