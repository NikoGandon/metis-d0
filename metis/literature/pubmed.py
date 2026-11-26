"""Collecteur PubMed (notices et abstracts) + PMC Open Access (texte intégral JATS).

Le client httpx est injecté : en test, un transport rejoue des réponses figées.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from metis import config
from metis.literature.textclean import construire_document
from metis.schemas import BaseLitterature, Document, FormatSource


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


class CollecteurPubMed:
    base = BaseLitterature.PUBMED

    def __init__(self, http: httpx.Client, pause_s: float = config.PAUSE_NCBI_S) -> None:
        self._http = http
        self._pause_s = pause_s

    def _get(self, outil: str, params: dict[str, str]) -> httpx.Response:
        complets = {**params, "tool": "metis"}
        email = config.ncbi_email()
        if email:
            complets["email"] = email
        reponse = self._http.get(f"{config.EUTILS_URL}/{outil}", params=complets)
        reponse.raise_for_status()
        if self._pause_s:
            time.sleep(self._pause_s)
        return reponse

    def rechercher(self, requete: str, max_resultats: int) -> tuple[str, ...]:
        """PMIDs dans l'ordre de pertinence PubMed."""
        donnees: Any = self._get(
            "esearch.fcgi", {"db": "pubmed", "term": requete, "retmax": str(max_resultats), "retmode": "json"}
        ).json()
        return tuple(str(pmid) for pmid in donnees["esearchresult"]["idlist"])

    def notice(self, pmid: str) -> tuple[Document, str | None]:
        """Notice PubMed (titre + abstract) et PMCID éventuel."""
        xml = self._get("efetch.fcgi", {"db": "pubmed", "id": pmid, "retmode": "xml"}).text
        pmcid = None
        # Seule la liste d'identifiants de l'article lui-même (PubmedData/ArticleIdList),
        # pas celles des références citées (ReferenceList/Reference/ArticleIdList).
        for identifiant in ET.fromstring(xml).iterfind("./PubmedArticle/PubmedData/ArticleIdList/ArticleId"):
            if identifiant.get("IdType") == "pmc" and identifiant.text:
                pmcid = identifiant.text.strip()
                break
        return construire_document(f"pmid:{pmid}", FormatSource.PUBMED_XML, xml), pmcid

    def texte_integral(self, pmcid: str) -> Document | None:
        """Texte intégral JATS si PMC le distribue (Open Access) ; None sinon."""
        xml = self._get("efetch.fcgi", {"db": "pmc", "id": pmcid.removeprefix("PMC"), "retmode": "xml"}).text
        if not any(_local(e.tag) == "body" for e in ET.fromstring(xml).iter()):
            return None  # éditeur n'autorisant pas le texte intégral : front matter seul
        return construire_document(f"pmc:{pmcid}", FormatSource.PMC_XML, xml)

    def collecter(self, requete: str, max_resultats: int) -> tuple[Document, ...]:
        documents: list[Document] = []
        for pmid in self.rechercher(requete, max_resultats):
            notice, pmcid = self.notice(pmid)
            documents.append(notice)
            if pmcid:
                integral = self.texte_integral(pmcid)
                if integral is not None:
                    documents.append(integral)
        return tuple(documents)
