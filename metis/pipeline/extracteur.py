"""Juge extracteur : document + QuestionD0 -> claims EXTRACTED et candidats d'équation.

Un appel par document (ou par segment d'un document long). L'extracteur ne voit
jamais les statuts ; ses citations sont ensuite vérifiées par verify.py.
"""

from __future__ import annotations

from collections.abc import Iterable

from metis import config
from metis.literature.verify import normaliser_citation
from metis.llm import ClientLLM, RunLedger, SortieLLMInvalideError, appeler_structure
from metis.schemas import (
    CandidatEquation,
    Claim,
    Document,
    Incident,
    Message,
    QuestionD0,
    RoleLLM,
    SortieExtracteur,
)

SYSTEME = """Tu es l'extracteur d'un audit documentaire scientifique fail-closed.

On te donne une sous-question de recherche et UN document (identifié par son source_id).
Tu dois relever les passages du document qui répondent directement à la sous-question.

Règles impératives :
1. Chaque citation_verbatim est un passage CONTIGU copié caractère pour caractère depuis
   le document : pas de reformulation, pas de traduction, pas de coupe interne, pas de
   fusion de deux phrases éloignées. Au moins 40 caractères.
2. source_id est recopié exactement tel qu'il est donné.
3. Ne relève que des passages pertinents pour la sous-question, l'espèce cible, la
   population et le mesurande ; n'invente rien. S'il n'y a rien de pertinent, renvoie des listes vides.
4. N'écarte pas un passage parce qu'il te semble faible : d'autres étapes le jugeront.
5. equations : seulement si le document contient une équation mécaniste explicite liée
   au mesurande. citation_equation recopie l'équation ET le texte qui l'entoure
   immédiatement (au moins 40 caractères, contigu, verbatim). observable_nom et
   observable_unite décrivent la grandeur calculée par l'équation ; laisse-les vides si
   le texte ne permet pas de les établir.
"""


def decouper(texte: str) -> list[str]:
    """Découpe en segments de MAX_CARACTERES_PAR_APPEL avec chevauchement (rien n'est omis)."""
    taille, recouvrement = config.MAX_CARACTERES_PAR_APPEL, config.CHEVAUCHEMENT_SEGMENTS
    if len(texte) <= taille:
        return [texte]
    segments, debut = [], 0
    while debut < len(texte):
        segments.append(texte[debut : debut + taille])
        if debut + taille >= len(texte):
            break
        debut += taille - recouvrement
    return segments


def _messages(question: QuestionD0, document: Document, segment: str, i: int, n: int) -> tuple[Message, ...]:
    entete = f"source_id : {document.source_id}"
    if n > 1:
        entete += f" (segment {i + 1}/{n} ; les segments se chevauchent de {config.CHEVAUCHEMENT_SEGMENTS} caractères)"
    return (
        Message(role="system", content=SYSTEME),
        Message(
            role="user",
            content=(
                f"QuestionD0 :\n{question.model_dump_json(indent=2)}\n\n"
                f"Document — {entete}\n<<<DEBUT\n{segment}\nFIN>>>"
            ),
        ),
    )


def extraire(
    question: QuestionD0,
    documents: Iterable[Document],
    client: ClientLLM,
    ledger: RunLedger,
    numero_initial: int = 1,
) -> tuple[tuple[Claim, ...], tuple[CandidatEquation, ...], tuple[Incident, ...]]:
    claims: list[Claim] = []
    equations: list[CandidatEquation] = []
    incidents: list[Incident] = []
    vus: set[tuple[str, str]] = set()
    numero = numero_initial
    for document in sorted(documents, key=lambda d: d.source_id):
        segments = decouper(document.texte_nettoye)
        for i, segment in enumerate(segments):
            messages = _messages(question, document, segment, i, len(segments))
            try:
                sortie = appeler_structure(client, ledger, RoleLLM.EXTRACTEUR, messages, SortieExtracteur)
            except SortieLLMInvalideError as echec:
                incidents.append(Incident(etape="extracteur", cible=document.source_id, motif=echec.motif))
                continue
            if len(sortie.claims) > config.MAX_CLAIMS_PAR_APPEL:
                motif = f"{len(sortie.claims)} claims proposés, {config.MAX_CLAIMS_PAR_APPEL} retenus"
                incidents.append(Incident(etape="extracteur", cible=document.source_id, motif=motif))
            for propose in sortie.claims[: config.MAX_CLAIMS_PAR_APPEL]:
                cle = (propose.source_id, normaliser_citation(propose.citation_verbatim))
                if cle in vus:
                    continue
                vus.add(cle)
                claims.append(Claim.extrait(f"c{numero:03d}", propose.source_id, propose.citation_verbatim))
                numero += 1
            for eq in sortie.equations[: config.MAX_EQUATIONS_PAR_APPEL]:
                candidat = CandidatEquation(
                    source_id=eq.source_id,
                    citation_equation=eq.citation_equation,
                    observable_nom=eq.observable_nom,
                    observable_unite=eq.observable_unite,
                )
                if candidat not in equations:
                    equations.append(candidat)
    return tuple(claims), tuple(equations), tuple(incidents)
