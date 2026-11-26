"""Réponses HTTP figées pour les collecteurs : rejeu, et enregistrement explicite.

Chaque réponse est un fichier <sha256(méthode + URL)>.json dans un dossier de corpus.
TransportFige rejoue ; une URL absente lève RequeteNonFigeeError, sauf si
enregistrer=True (METIS_CASSETTE=record, réseau débloqué) : elle est alors récupérée
en direct et figée. Voir aussi figer_corpus.py pour figer hors pytest.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx


class RequeteNonFigeeError(RuntimeError):
    """Le collecteur a demandé une URL absente du corpus figé."""


def cle_requete(requete: httpx.Request) -> str:
    return hashlib.sha256(f"{requete.method} {requete.url}".encode()).hexdigest()


def _reponse(donnees: dict[str, object], requete: httpx.Request) -> httpx.Response:
    assert isinstance(donnees["status"], int) and isinstance(donnees["corps_b64"], str)
    entetes = {"content-type": str(donnees["content_type"])}
    if donnees.get("location"):
        entetes["location"] = str(donnees["location"])  # redirections suivies par httpx
    return httpx.Response(
        donnees["status"], headers=entetes, content=base64.b64decode(donnees["corps_b64"]), request=requete
    )


class TransportFige(httpx.BaseTransport):
    def __init__(self, dossier: Path, enregistrer: bool = False) -> None:
        self.dossier = dossier
        self.enregistrer = enregistrer
        self.urls_servies: list[str] = []
        self._reel: httpx.HTTPTransport | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        chemin = self.dossier / f"{cle_requete(request)}.json"
        if not chemin.exists():
            if not self.enregistrer:
                raise RequeteNonFigeeError(f"non figée : {request.method} {request.url}")
            self._figer(request, chemin)
        self.urls_servies.append(str(request.url))
        return _reponse(json.loads(chemin.read_text(encoding="utf-8")), request)

    def _figer(self, request: httpx.Request, chemin: Path) -> None:
        if self._reel is None:
            self._reel = httpx.HTTPTransport()
        reponse = self._reel.handle_request(request)
        corps = reponse.read()  # décodé : content-encoding n'est pas conservé
        enregistrement = {
            "methode": request.method,
            "url": str(request.url),
            "status": reponse.status_code,
            "content_type": reponse.headers.get("content-type", "application/octet-stream"),
            "location": reponse.headers.get("location"),
            "corps_b64": base64.b64encode(corps).decode("ascii"),
        }
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(json.dumps(enregistrement, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def client_fige(dossier: Path, enregistrer: bool = False) -> httpx.Client:
    return httpx.Client(transport=TransportFige(dossier, enregistrer), timeout=60)
