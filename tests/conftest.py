"""Fixtures communes : aucun appel réseau dans tests/, cassettes en lecture seule.

Seules exceptions : METIS_CASSETTE=record ou record_missing, utilisés explicitement
pour (ré)enregistrer des cassettes (voir README).
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import httpx
import pytest


class ReseauInterditError(RuntimeError):
    pass


def _interdit(*_args: Any, **_kwargs: Any) -> Any:
    raise ReseauInterditError("appel réseau non cassetté tenté dans tests/")


_connect_origine = socket.socket.connect


def _connect_filtre(self: socket.socket, adresse: Any) -> None:
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _interdit()
    _connect_origine(self, adresse)


@pytest.fixture(autouse=True)
def _sans_reseau(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if os.environ.get("METIS_CASSETTE") in ("record", "record_missing"):
        yield
        return
    monkeypatch.setenv("METIS_CASSETTE", "replay")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _interdit)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _interdit)
    monkeypatch.setattr(socket.socket, "connect", _connect_filtre)
    yield
