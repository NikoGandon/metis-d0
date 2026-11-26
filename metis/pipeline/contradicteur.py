"""Juge contradicteur : cherche activement des réfutations d'un claim ESTABLISHED.

Deux appels distincts : formuler au plus 3 requêtes adverses (exécutées par les
collecteurs, hors de ce module), puis proposer des passages réfutants dans chaque
document reçu. Les propositions deviennent des PreuveAdverse CANDIDATE, qui passent
ensuite exactement les mêmes portes qu'un claim.
"""

from __future__ import annotations

import json

from metis import config
from metis.llm import ClientLLM, RunLedger, appeler_structure
from metis.pipeline.extracteur import decouper
from metis.schemas import (
    Claim,
    Document,
    Message,
    PreuveProposee,
    QuestionD0,
    RequeteAdverse,
    RoleLLM,
    SortiePreuves,
    SortieRequetesAdverses,
)

SYSTEME_REQUETES = f"""Tu es le contradicteur d'un audit documentaire scientifique fail-closed.

On te donne une QuestionD0 et un claim jugé établi. Formule au plus
{config.MAX_REQUETES_ADVERSES} requêtes de recherche bibliographique destinées à trouver
des publications qui le RÉFUTENT : résultats nuls, effet opposé, échec de réplication,
effets délétères sur le même mesurande, dans la même espèce et la même population.
base : « pubmed » (syntaxe PubMed) ou « arxiv » (search_query arXiv, ex. all:terme).
Des requêtes courtes et précises valent mieux que des requêtes exhaustives."""

SYSTEME_PREUVES = f"""Tu es le contradicteur d'un audit documentaire scientifique fail-closed.

On te donne une QuestionD0, un claim jugé établi et UN document trouvé par une requête
adverse. Relève au plus {config.MAX_PREUVES_PAR_DOCUMENT} passages de CE document qui
contredisent directement le claim (même espèce, même population, même intervention,
même mesurande).

Règles : citation_verbatim est un passage contigu copié caractère pour caractère, au
moins 40 caractères, sans reformulation ni traduction ; source_id est recopié tel quel.
Un simple partage de mots-clés n'est pas une réfutation. Si rien ne réfute le claim,
renvoie une liste vide."""


def _claim_json(question: QuestionD0, claim: Claim) -> str:
    return json.dumps(
        {
            "question": question.model_dump(mode="json"),
            "claim": {"source_id": claim.source_id, "citation_verbatim": claim.citation_verbatim},
        },
        ensure_ascii=False,
        indent=2,
    )


def formuler_requetes(
    claim: Claim, question: QuestionD0, client: ClientLLM, ledger: RunLedger
) -> tuple[RequeteAdverse, ...]:
    """Lève SortieLLMInvalideError ; l'appelant journalise l'incident (fail-closed)."""
    messages = (
        Message(role="system", content=SYSTEME_REQUETES),
        Message(role="user", content=_claim_json(question, claim)),
    )
    sortie = appeler_structure(client, ledger, RoleLLM.CONTRADICTEUR, messages, SortieRequetesAdverses)
    return sortie.requetes[: config.MAX_REQUETES_ADVERSES]


def proposer_preuves(
    claim: Claim, question: QuestionD0, document: Document, client: ClientLLM, ledger: RunLedger
) -> tuple[PreuveProposee, ...]:
    """Un appel par segment du document. Lève SortieLLMInvalideError ; l'appelant
    journalise l'incident (fail-closed)."""
    preuves: list[PreuveProposee] = []
    segments = decouper(document.texte_nettoye)
    for i, segment in enumerate(segments):
        entete = f"source_id : {document.source_id}"
        if len(segments) > 1:
            entete += f" (segment {i + 1}/{len(segments)})"
        messages = (
            Message(role="system", content=SYSTEME_PREUVES),
            Message(
                role="user",
                content=f"{_claim_json(question, claim)}\n\nDocument — {entete}\n<<<DEBUT\n{segment}\nFIN>>>",
            ),
        )
        sortie = appeler_structure(client, ledger, RoleLLM.CONTRADICTEUR, messages, SortiePreuves)
        preuves.extend(p for p in sortie.preuves if p not in preuves)
    return tuple(preuves[: config.MAX_PREUVES_PAR_DOCUMENT])
