"""Gate verify.py : citation verbatim, table d'offsets, fenêtre de contexte (0 LLM).

Normalisation (NFKC, casse, tirets, guillemets, espaces insécables, espaces
multiples) appliquée grappe par grappe : une grappe est un caractère de base suivi
de ses marques combinantes (catégorie Unicode M*). Chaque caractère normalisé
garde la grappe d'origine qui l'a produit, d'où la table normalise[i] -> original.
Comme la même fonction s'applique grappe par grappe au source et à la citation,
toute sous-chaîne du source découpée aux bornes de grappes se retrouve telle
quelle dans le source normalisé. C'est le seul traitement commun aux deux côtés :
textclean.py ne s'applique qu'au document source.

Occurrences multiples : jusqu'à 3 fenêtres sont conservées et transmises à
l'auditeur, dont le statut le plus défavorable l'emporte.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from metis.schemas import (
    MAX_OCCURRENCES_CONSERVEES,
    Claim,
    Document,
    Occurrence,
    Offsets,
    Porte,
    PreuveAdverse,
    StatutClaim,
    StatutPreuve,
)

journal = logging.getLogger(__name__)

LONGUEUR_MIN_CITATION = 40
DEMI_FENETRE = 250

_SUPPRIMES = "­​‌‍⁠﻿"  # trait d'union conditionnel, largeur nulle
_TIRETS = "‐‑‒–—―−﹘﹣－"
_APOSTROPHES = "‘’‚‛′ʼ"
_GUILLEMETS = "“”„‟″«»"
_TABLE = str.maketrans(
    {
        **{c: None for c in _SUPPRIMES},
        **{c: "-" for c in _TIRETS},
        **{c: "'" for c in _APOSTROPHES},
        **{c: '"' for c in _GUILLEMETS},
    }
)


@dataclass(frozen=True)
class TexteNormalise:
    """texte[i] provient de la grappe original[debuts[i]:fins[i]]."""

    texte: str
    debuts: tuple[int, ...]
    fins: tuple[int, ...]


def _normaliser_grappe(grappe: str) -> str:
    plie = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", grappe).casefold())
    return plie.translate(_TABLE)


def normaliser(texte: str) -> TexteNormalise:
    sortie: list[str] = []
    debuts: list[int] = []
    fins: list[int] = []
    i, n = 0, len(texte)
    while i < n:
        j = i + 1
        while j < n and unicodedata.category(texte[j]).startswith("M"):
            j += 1
        for c in _normaliser_grappe(texte[i:j]):
            if c.isspace():
                if sortie and sortie[-1] == " ":
                    fins[-1] = j  # l'espace fusionnée couvre aussi cette grappe
                    continue
                c = " "
            sortie.append(c)
            debuts.append(i)
            fins.append(j)
        i = j
    return TexteNormalise("".join(sortie), tuple(debuts), tuple(fins))


def normaliser_citation(citation: str) -> str:
    return normaliser(citation).texte.strip(" ")


@lru_cache(maxsize=64)
def _normaliser_source(texte: str) -> TexteNormalise:
    return normaliser(texte)


def _aux_bornes_de_grappe(source: TexteNormalise, a: int, b: int) -> bool:
    """Le motif [a, b) ne coupe pas la sortie d'une grappe (hors espaces)."""
    k = a - 1
    while k >= 0 and source.debuts[k] == source.debuts[a]:
        if source.texte[k] != " ":
            return False
        k -= 1
    k = b
    while k < len(source.texte) and source.debuts[k] == source.debuts[b - 1]:
        if source.texte[k] != " ":
            return False
        k += 1
    return True


def _occurrences(source: TexteNormalise, motif: str) -> list[int]:
    positions: list[int] = []
    a = source.texte.find(motif)
    while a != -1:
        if _aux_bornes_de_grappe(source, a, a + len(motif)):
            positions.append(a)
        a = source.texte.find(motif, a + 1)
    return positions


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


class MotifVerify(StrEnum):
    SOURCE_HORS_CORPUS = "source_hors_corpus"
    CITATION_TROP_COURTE = "citation_absente_ou_trop_courte"
    CITATION_NON_VERBATIM = "citation_non_verbatim"


@dataclass(frozen=True)
class Ancrage:
    """Citation acceptée. occurrences : au plus 3, la première est retenue pour les
    offsets du claim ; toutes les fenêtres partent à l'auditeur."""

    occurrences: tuple[Occurrence, ...]
    nombre_occurrences: int  # total des occurrences alignées dans le document

    @property
    def offsets(self) -> Offsets:
        return self.occurrences[0].offsets

    @property
    def context_window(self) -> str:
        return self.occurrences[0].context_window


ResultatVerify = Ancrage | MotifVerify


def _occurrence(source: TexteNormalise, original: str, a: int, longueur: int) -> Occurrence:
    debut, fin = source.debuts[a], source.fins[a + longueur - 1]
    fenetre = original[max(0, debut - DEMI_FENETRE) : min(len(original), fin + DEMI_FENETRE)]
    return Occurrence(offsets=Offsets(debut=debut, fin=fin), context_window=fenetre)


def verifier_citation(source_id: str, citation: str, corpus: Mapping[str, Document]) -> ResultatVerify:
    """corpus : uniquement les documents téléchargés dans le run courant."""
    document = corpus.get(source_id)
    if document is None:
        return MotifVerify.SOURCE_HORS_CORPUS
    motif = normaliser_citation(citation)
    if len(motif) < LONGUEUR_MIN_CITATION:
        return MotifVerify.CITATION_TROP_COURTE
    source = _normaliser_source(document.texte_nettoye)
    positions = _occurrences(source, motif)
    if not positions:
        return MotifVerify.CITATION_NON_VERBATIM
    if len(positions) > 1:
        journal.info("verify : %d occurrences dans %s", len(positions), source_id)
    occurrences = tuple(
        _occurrence(source, document.texte_nettoye, a, len(motif))
        for a in positions[:MAX_OCCURRENCES_CONSERVEES]
    )
    return Ancrage(occurrences, len(positions))


def verifier_claim(claim: Claim, corpus: Mapping[str, Document]) -> tuple[Claim, ResultatVerify]:
    resultat = verifier_citation(claim.source_id, claim.citation_verbatim, corpus)
    if isinstance(resultat, MotifVerify):
        return claim.transitionner(StatutClaim.REJECTED_VERBATIM, Porte.VERIFY, resultat.value), resultat
    return claim.avec_ancrage(resultat.occurrences, resultat.nombre_occurrences), resultat


def verifier_preuve(
    preuve: PreuveAdverse, corpus: Mapping[str, Document]
) -> tuple[PreuveAdverse, ResultatVerify]:
    resultat = verifier_citation(preuve.source_id, preuve.citation_verbatim, corpus)
    if isinstance(resultat, MotifVerify):
        return preuve.transitionner(StatutPreuve.REJECTED_VERBATIM, Porte.VERIFY, resultat.value), resultat
    return preuve.avec_ancrage(resultat.occurrences, resultat.nombre_occurrences), resultat
