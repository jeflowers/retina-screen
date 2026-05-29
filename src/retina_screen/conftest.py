"""Test fixtures. Autouse fixture blocks all socket access — tests must be hermetic."""

import socket
from collections.abc import Iterator
from typing import Any, NoReturn

import pytest

__all__: list[str] = []


class _NetworkBlockedError(RuntimeError):
    """Raised when a test attempts a real network call."""


def _blocked(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise _NetworkBlockedError("Network access is blocked in tests. Inject a fake client instead.")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block real socket connections across every test.

    Real boto3 clients (S3, DynamoDB, etc.) ultimately use sockets; this fixture
    catches the case where a test forgot to inject a fake. UNIX-domain sockets
    are left alone so pytest's own machinery (e.g. capture) keeps working.
    """
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield
