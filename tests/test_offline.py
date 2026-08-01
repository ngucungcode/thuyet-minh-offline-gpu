from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_offline_guard_blocks_dns_and_inet_without_poisoning_test_process() -> None:
    script = r'''
import socket
from dub_server.offline import OfflineNetworkError, install_offline_network_guard

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.bind(("127.0.0.1", 0))
listener.listen(1)
allowed_port = listener.getsockname()[1]

install_offline_network_guard(allowed_loopback_ports={allowed_port})
install_offline_network_guard(allowed_loopback_ports={allowed_port})

socket.getaddrinfo("127.0.0.1", allowed_port)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", allowed_port))
accepted, _ = listener.accept()
accepted.close()
client.close()
listener.close()

blocked = 0
for operation in (
    lambda: socket.getaddrinfo("example.com", 443),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("198.51.100.1", 443)),
    lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM).connect(("2001:db8::1", 443)),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", allowed_port + 1)),
):
    try:
        operation()
    except OfflineNetworkError:
        blocked += 1

assert blocked == 4, blocked
print("offline-guard-ok")
'''
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline-guard-ok"
