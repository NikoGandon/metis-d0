"""Collecteur arXiv : résumés (API Atom) + source LaTeX si disponible.

La source e-print est lue en mémoire (tar.gz ou .gz) ; seuls les fichiers .tex
sont décodés, rien n'est écrit sur disque ni exécuté.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import time
import xml.etree.ElementTree as ET

import httpx

from metis import config
from metis.literature.textclean import construire_document
from metis.schemas import BaseLitterature, Document, FormatSource

_ATOM = "{http://www.w3.org/2005/Atom}"


class CollecteurArxiv:
    base = BaseLitterature.ARXIV

    def __init__(self, http: httpx.Client, pause_s: float = config.PAUSE_ARXIV_S) -> None:
        self._http = http
        self._pause_s = pause_s

    def _get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        reponse = self._http.get(url, params=params, follow_redirects=True)
        reponse.raise_for_status()
        if self._pause_s:
            time.sleep(self._pause_s)
        return reponse

    def rechercher(self, requete: str, max_resultats: int) -> tuple[Document, ...]:
        """Résumés ; requete est une search_query arXiv (ex. « all:gompertz AND all:tumor »)."""
        flux = self._get(
            config.ARXIV_API_URL, {"search_query": requete, "start": "0", "max_results": str(max_resultats)}
        ).text
        documents = []
        for entree in ET.fromstring(flux).iter(f"{_ATOM}entry"):
            identifiant = (entree.findtext(f"{_ATOM}id") or "").rsplit("/abs/", 1)[-1].strip()
            if not identifiant:
                continue
            brut = ET.tostring(entree, encoding="unicode")
            documents.append(construire_document(f"arxiv:{identifiant}", FormatSource.ARXIV_ABSTRACT, brut))
        return tuple(documents)

    def source_latex(self, identifiant: str) -> Document | None:
        reponse = self._get(f"{config.ARXIV_EPRINT_URL}/{identifiant}")
        contenu = reponse.content
        if len(contenu) > config.MAX_OCTETS_SOURCE_LATEX:
            return None
        texte = _extraire_tex(contenu)
        if texte is None:
            return None
        return construire_document(f"arxiv:{identifiant}:tex", FormatSource.ARXIV_LATEX, texte)

    def collecter(self, requete: str, max_resultats: int) -> tuple[Document, ...]:
        documents: list[Document] = []
        for resume in self.rechercher(requete, max_resultats):
            documents.append(resume)
            latex = self.source_latex(resume.source_id.removeprefix("arxiv:"))
            if latex is not None:
                documents.append(latex)
        return tuple(documents)


def _extraire_tex(contenu: bytes) -> str | None:
    """tar(.gz) -> fichiers .tex concaténés par nom ; .gz seul -> le fichier ; PDF -> None."""
    try:
        with tarfile.open(fileobj=io.BytesIO(contenu), mode="r:*") as archive:
            parties = []
            total = 0
            for membre in sorted(archive.getmembers(), key=lambda m: m.name):
                if not (membre.isfile() and membre.name.endswith(".tex")):
                    continue
                total += membre.size
                if total > config.MAX_OCTETS_SOURCE_LATEX:
                    return None
                fichier = archive.extractfile(membre)
                if fichier is not None:
                    tex = fichier.read().decode("utf-8", errors="replace")
                    parties.append(f"%% FICHIER {membre.name}\n{tex}")
            return "\n".join(parties) or None
    except tarfile.TarError:
        pass
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(contenu)) as flux:
            brut = flux.read(config.MAX_OCTETS_SOURCE_LATEX + 1)  # borne la décompression
    except (OSError, EOFError):
        return None
    if len(brut) > config.MAX_OCTETS_SOURCE_LATEX or brut.startswith(b"%PDF"):
        return None
    return brut.decode("utf-8", errors="replace")
