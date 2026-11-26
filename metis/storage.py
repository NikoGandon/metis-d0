"""Sérialisation JSON canonique et écriture atomique.

Aucune dépendance vers le reste de metis : schemas.py s'appuie sur json_canonique
pour le contrat_hash.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _en_json(objet: Any) -> Any:
    return objet.model_dump(mode="json") if isinstance(objet, BaseModel) else objet


def json_canonique(objet: Any) -> bytes:
    """Forme compacte, clés triées, UTF-8 ; NaN/Inf refusés. Sert aux hachages."""
    return json.dumps(
        _en_json(objet), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _json_fichier(objet: Any) -> bytes:
    """Même canonicité que json_canonique, indentée pour la lecture humaine."""
    texte = json.dumps(_en_json(objet), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    return (texte + "\n").encode("utf-8")


def ecrire_json_atomique(chemin: Path, objet: Any) -> None:
    """Écrit objet dans chemin sans jamais exposer un fichier partiel.

    Fichier temporaire *.tmp dans le même répertoire, fsync, os.replace, puis fsync
    du répertoire. Une erreur avant os.replace laisse l'ancien fichier intact.
    """
    _ecrire_atomique(chemin, _json_fichier(objet))  # sérialise avant de toucher au disque


def ecrire_texte_atomique(chemin: Path, texte: str) -> None:
    _ecrire_atomique(chemin, texte.encode("utf-8"))


def _ecrire_atomique(chemin: Path, contenu: bytes) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=chemin.parent, prefix=f".{chemin.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contenu)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, chemin)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_repertoire(chemin.parent)


def _fsync_repertoire(repertoire: Path) -> None:
    fd = os.open(repertoire, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def lire_json(chemin: Path) -> Any:
    return json.loads(chemin.read_text(encoding="utf-8"))
