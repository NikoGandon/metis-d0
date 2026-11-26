"""Routeur : gate mécanique (0 LLM) puis juge LLM qui ne peut que confirmer ou rétrograder.

Gate, dans l'ordre :
1. au moins un claim ESTABLISHED non contredit et dont la recherche de réfutations
   s'est déroulée sans incident -> sinon refus / claims_insuffisants ;
2. au moins un candidat d'équation -> sinon refus / pas_d_equation ;
3. pour chaque candidat : papier_cible dans le corpus du run et citation_equation
   passe verify.py (equation_non_verbatim), contient « = » ou une commande LaTeX
   d'équation (pas_d_equation), observable nommé et unité non vide (pas_d_observable).
   Le premier candidat qui passe tout va au juge ; sinon, le motif du candidat allé
   le plus loin.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from metis.literature.verify import Ancrage, verifier_citation
from metis.llm import ClientLLM, RunLedger, SortieLLMInvalideError, appeler_structure
from metis.schemas import (
    CandidatEquation,
    Claim,
    Document,
    Grandeur,
    Incident,
    Message,
    MotifRefus,
    QuestionD0,
    RoleLLM,
    SortieRouteur,
    StatutClaim,
    Verdict,
    VerdictRoutage,
)

_RE_COMMANDE_EQUATION = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|exp|ln|log|sum|prod|int|partial|cdot|times|left|right"
    r"|begin\{(?:equation|align|eqnarray|gather|multline)\*?\})"
)

SYSTEME = """Tu es le juge du routeur d'un audit documentaire fail-closed.

Une gate mécanique a déjà vérifié qu'une équation est citée verbatim dans un papier du
corpus et qu'un observable est nommé. Tu dois seulement CONFIRMER ou REFUSER :
- confirmer si l'équation citée est une loi mécaniste (issue d'un mécanisme : cinétique,
  bilan, dynamique) et non une régression descriptive ajustée sur des données, ET si ses
  paramètres sont donnés ou mesurables à partir du texte accessible, ET si la sous-question
  porte bien sur une grandeur que cette équation permet de simuler ;
- sinon refuser avec un motif : pas_d_equation (régression descriptive, pas de loi
  mécaniste), parametres_inaccessibles, question_documentaire (la sous-question n'appelle
  pas de simulation), ou juge_refuse (autre raison).
motif_refus vaut null si tu confirmes. En cas de doute, refuse."""

_RANG_GATE = {
    MotifRefus.EQUATION_NON_VERBATIM: 0,
    MotifRefus.PAS_D_EQUATION: 1,
    MotifRefus.PAS_D_OBSERVABLE: 2,
}


def _refus(motif: MotifRefus) -> VerdictRoutage:
    return VerdictRoutage(verdict=Verdict.REFUS, motif_refus=motif)


def claims_eligibles(claims: Sequence[Claim], incidents: Sequence[Incident]) -> tuple[Claim, ...]:
    sans_refutation_complete = {i.cible for i in incidents if i.etape == "contradicteur"}
    return tuple(
        c for c in claims if c.statut is StatutClaim.ESTABLISHED and c.claim_id not in sans_refutation_complete
    )


def _controler_candidat(candidat: CandidatEquation, corpus: Mapping[str, Document]) -> MotifRefus | Ancrage:
    ancrage = verifier_citation(candidat.source_id, candidat.citation_equation, corpus)
    if not isinstance(ancrage, Ancrage):
        return MotifRefus.EQUATION_NON_VERBATIM
    if "=" not in candidat.citation_equation and not _RE_COMMANDE_EQUATION.search(candidat.citation_equation):
        return MotifRefus.PAS_D_EQUATION
    if not candidat.observable_nom.strip() or not candidat.observable_unite.strip():
        return MotifRefus.PAS_D_OBSERVABLE
    return ancrage


def gate(
    claims: Sequence[Claim],
    incidents: Sequence[Incident],
    candidats: Sequence[CandidatEquation],
    corpus: Mapping[str, Document],
) -> VerdictRoutage | tuple[CandidatEquation, Ancrage]:
    """Refus mécanique, ou le candidat retenu pour le juge."""
    if not claims_eligibles(claims, incidents):
        return _refus(MotifRefus.CLAIMS_INSUFFISANTS)
    if not candidats:
        return _refus(MotifRefus.PAS_D_EQUATION)
    echecs: list[MotifRefus] = []
    for candidat in candidats:
        resultat = _controler_candidat(candidat, corpus)
        if isinstance(resultat, Ancrage):
            return candidat, resultat
        echecs.append(resultat)
    return _refus(max(echecs, key=lambda m: _RANG_GATE[m]))


def router(
    question: QuestionD0,
    claims: Sequence[Claim],
    incidents: Sequence[Incident],
    candidats: Sequence[CandidatEquation],
    corpus: Mapping[str, Document],
    client: ClientLLM,
    ledger: RunLedger,
) -> VerdictRoutage:
    resultat = gate(claims, incidents, candidats, corpus)
    if isinstance(resultat, VerdictRoutage):
        return resultat
    candidat, ancrage = resultat
    charge = {
        "question": question.model_dump(mode="json"),
        "claims_etablis": [
            {"source_id": c.source_id, "citation_verbatim": c.citation_verbatim}
            for c in claims_eligibles(claims, incidents)
        ],
        "equation": {
            "papier_cible": candidat.source_id,
            "citation_equation": candidat.citation_equation,
            "fenetre": ancrage.context_window,
            "observable": {"nom": candidat.observable_nom, "unite": candidat.observable_unite},
        },
    }
    messages = (
        Message(role="system", content=SYSTEME),
        Message(role="user", content=json.dumps(charge, ensure_ascii=False, indent=2)),
    )
    try:
        sortie = appeler_structure(client, ledger, RoleLLM.ROUTEUR, messages, SortieRouteur)
    except SortieLLMInvalideError:
        return _refus(MotifRefus.JUGE_REFUSE)
    if sortie.decision == "refuser":
        return _refus(MotifRefus(sortie.motif_refus) if sortie.motif_refus else MotifRefus.JUGE_REFUSE)
    return VerdictRoutage(
        verdict=Verdict.SIMULABLE,
        papier_cible=candidat.source_id,
        citation_equation=candidat.citation_equation,
        observable=Grandeur(nom=candidat.observable_nom, unite=candidat.observable_unite),
    )
