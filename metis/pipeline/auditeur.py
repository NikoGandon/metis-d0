"""Juge auditeur : statue sur un Claim (ou une PreuveAdverse) déjà passé par les gates.

Contexte transmis (config.SEUIL_DOCUMENT_ENTIER) :
- document court (abstract) : le document entier, chaque occurrence retenue de la
  citation balisée <<<…>>>, et une seule évaluation ;
- document long (texte intégral) : les fenêtres conservées par verify.py (au plus 3),
  une évaluation par fenêtre, plus le résumé du document pour la population et le plan
  d'étude. Le harnais retient l'évaluation la plus défavorable.

Le juge ne peut que rejeter ou dégrader ; le code applique ensuite, sans exception :
- un axe autre que la temporalité (espèce, mesurande, population, intervention) non
  conforme ou indéterminé -> REJECTED_SEMANTIC ;
- CLAIMED_UNPROVEN seulement si l'axe temporalité est non conforme (et lui seul) ;
- ESTABLISHED seulement si les six axes sont conformes, et si la citation porte
  is_future_or_protocol, elle doit contenir un résultat chiffré
  (config.MOTIF_RESULTAT_CHIFFRE), sinon CLAIMED_UNPROVEN ;
- une preuve adverse n'est CONFIRMED que si ses six axes sont conformes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from metis import config
from metis.literature.textclean import extraire_resume
from metis.llm import ClientLLM, RunLedger, SortieLLMInvalideError, appeler_structure
from metis.schemas import (
    AxeJugement,
    Claim,
    Document,
    EvaluationFenetre,
    EvaluationFenetrePreuve,
    Message,
    Porte,
    PreuveAdverse,
    QuestionD0,
    RoleLLM,
    SortieAuditeur,
    SortieAuditeurPreuve,
    StatutClaim,
    StatutPreuve,
)

TEMPORALITE = "temporalite_et_statut_de_preuve"
BALISE_OUVRANTE, BALISE_FERMANTE = "<<<", ">>>"

_REGLES_AXES = """Axes, chacun jugé « conforme », « non_conforme » ou « indetermine » avec une
justification d'une phrase tirée du contexte :
- espece_et_modele : le résultat porte-t-il sur l'espèce et le système demandés
  (ex. human_in_vivo : des humains vivants, pas des animaux, cellules ou simulations) ?
- temporalite_et_statut_de_preuve : est-ce un résultat observé, au passé, chiffré ou
  explicitement constaté ? Un protocole, un objectif, une phrase au futur, une
  hypothèse ou un essai enregistré sans résultat n'est pas une preuve.
- mesurande : le résultat porte-t-il sur le mesurande demandé (pas un mesurande voisin) ?
- population : la population étudiée correspond-elle au champ « population » de la
  QuestionD0 (âge, sexe, cadre de vie) ?
- intervention : l'intervention étudiée correspond-elle au champ « intervention » de la
  QuestionD0 (molécule, dose, rythme, cointerventions, comparateur) ?
« indetermine » si le contexte ne permet pas de vérifier l'axe. Lis tout le contexte :
une négation, une restriction ou une non-significativité près de la citation change son
sens."""

_REGLES_CONTEXTE = """Contextes : soit le document entier, où chaque occurrence de la citation est
balisée <<<…>>> (un seul contexte), soit des fenêtres autour de chaque occurrence
(numérotées à partir de 0), accompagnées du résumé du document ; utilise ce résumé
pour vérifier la population et l'intervention."""

SYSTEME_CLAIM = f"""Tu es l'auditeur d'un audit documentaire scientifique fail-closed.

On te donne une QuestionD0, une citation vérifiée verbatim, son contexte et des
drapeaux lexicaux mécaniques (indicatifs, non décisifs).

{_REGLES_CONTEXTE}

{_REGLES_AXES}

Pour CHAQUE contexte, rends une évaluation avec un statut :
- ESTABLISHED : les six axes sont conformes ;
- CLAIMED_UNPROVEN : seul l'axe temporalité n'est pas conforme (protocole, intention,
  futur, essai sans résultat rapporté) ;
- REJECTED_SEMANTIC : tout autre cas (hors sujet, négation, mesurande, espèce, modèle,
  population ou intervention différents ou non vérifiables, sur-interprétation).
Si is_future_or_protocol est vrai et qu'aucun résultat chiffré au passé ne figure dans
le contexte, le statut ne peut pas être ESTABLISHED. En cas de doute, ne choisis pas
ESTABLISHED. motif : une phrase."""

SYSTEME_PREUVE = f"""Tu es l'auditeur d'un audit documentaire scientifique fail-closed.

On te donne une QuestionD0, un claim établi et une PREUVE ADVERSE candidate, vérifiée
verbatim, avec son contexte et des drapeaux lexicaux mécaniques.

{_REGLES_CONTEXTE}

{_REGLES_AXES}
- force_de_refutation : la preuve contredit-elle DIRECTEMENT le claim, sur la même espèce,
  la même population, la même intervention et le même mesurande ? Un simple partage de
  mots-clés n'est pas une réfutation.

Pour CHAQUE contexte, rends une évaluation avec un statut :
- CONFIRMED : les six axes sont conformes ;
- REJECTED_SEMANTIC : sinon.
En cas de doute, REJECTED_SEMANTIC. motif : une phrase."""


# --------------------------------------------------------------------------- #
# Contexte
# --------------------------------------------------------------------------- #


def baliser(texte: str, bornes: Sequence[tuple[int, int]]) -> str:
    """Insère <<< >>> autour de chaque occurrence (bornes dans texte, sans chevauchement)."""
    morceaux, curseur = [], 0
    for debut, fin in sorted(bornes):
        if debut < curseur:
            continue  # occurrence chevauchant la précédente : déjà balisée
        morceaux += [texte[curseur:debut], BALISE_OUVRANTE, texte[debut:fin], BALISE_FERMANTE]
        curseur = fin
    morceaux.append(texte[curseur:])
    return "".join(morceaux)


def contextes(element: Claim | PreuveAdverse, document: Document) -> tuple[list[dict[str, object]], str | None]:
    """(contextes numérotés, résumé ou None) selon la longueur du document."""
    if element.source_id != document.source_id or element.offsets is None:
        raise ValueError("l'auditeur exige l'élément ancré dans son propre document")
    if len(document.texte_nettoye) < config.SEUIL_DOCUMENT_ENTIER:
        bornes = [(element.offsets.debut, element.offsets.fin)]
        bornes += [(o.offsets.debut, o.offsets.fin) for o in element.occurrences_supplementaires]
        entier = baliser(document.texte_nettoye, bornes)
        return [{"indice": 0, "type": "document_entier", "texte": entier}], None
    fenetres = [{"indice": i, "type": "fenetre", "texte": f} for i, f in enumerate(element.fenetres)]
    return fenetres, extraire_resume(document)


def _controle(n: int) -> Callable[[SortieAuditeur | SortieAuditeurPreuve], str | None]:
    def controle(sortie: SortieAuditeur | SortieAuditeurPreuve) -> str | None:
        indices = sorted(e.indice_contexte for e in sortie.evaluations)
        if indices != list(range(n)):
            return f"il faut exactement une évaluation par contexte, indices 0 à {n - 1} ; reçu {indices}"
        return None

    return controle


def _entree(
    question: QuestionD0,
    element: Claim | PreuveAdverse,
    liste: list[dict[str, object]],
    resume: str | None,
    extra: dict[str, object],
) -> str:
    charge = {
        "question": question.model_dump(mode="json"),
        **extra,
        "citation_verbatim": element.citation_verbatim,
        "source_id": element.source_id,
        "drapeaux": element.drapeaux.model_dump(mode="json") if element.drapeaux else None,
        "contextes": liste,
        "resume_du_document": resume,
    }
    return json.dumps(charge, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Règles de statut (code, pas juge)
# --------------------------------------------------------------------------- #

PREFIXE_AXES = "non conformes : "
_RANG_CLAIM = {StatutClaim.REJECTED_SEMANTIC: 0, StatutClaim.CLAIMED_UNPROVEN: 1, StatutClaim.ESTABLISHED: 2}
_RE_RESULTAT_CHIFFRE = re.compile(config.MOTIF_RESULTAT_CHIFFRE)
_RE_PUISSANCE = re.compile(config.MOTIF_PUISSANCE_EXCLUE, re.IGNORECASE)


def _motif(indice: int, axes: dict[str, AxeJugement], texte: str) -> str:
    """« contexte i [non conformes : a, b] : texte » ; le test red team lit les axes."""
    non_conformes = [nom for nom, axe in axes.items() if axe.verdict != "conforme"]
    return f"contexte {indice} [{PREFIXE_AXES}{', '.join(non_conformes) or 'aucun'}] : {texte}"


def axes_non_conformes(motif: str | None) -> frozenset[str]:
    if not motif or f"[{PREFIXE_AXES}" not in motif:
        return frozenset()
    liste = motif.split(f"[{PREFIXE_AXES}", 1)[1].split("]", 1)[0]
    return frozenset() if liste == "aucun" else frozenset(liste.split(", "))


def contient_resultat_chiffre(citation: str) -> bool:
    return _RE_RESULTAT_CHIFFRE.search(_RE_PUISSANCE.sub(" ", citation)) is not None


def _axes_claim(evaluation: EvaluationFenetre) -> dict[str, AxeJugement]:
    return {
        "espece_et_modele": evaluation.espece_et_modele,
        TEMPORALITE: evaluation.temporalite_et_statut_de_preuve,
        "mesurande": evaluation.mesurande,
        "population": evaluation.population,
        "intervention": evaluation.intervention,
    }


def statut_claim(evaluation: EvaluationFenetre, claim: Claim) -> tuple[StatutClaim, str]:
    """Statut retenu pour un contexte : le plus défavorable entre le juge et les règles."""
    axes = _axes_claim(evaluation)
    motif = _motif(evaluation.indice_contexte, axes, evaluation.motif)
    non_conformes = {nom for nom, axe in axes.items() if axe.verdict != "conforme"}
    juge = StatutClaim(evaluation.statut)
    if juge is StatutClaim.REJECTED_SEMANTIC:
        return juge, motif
    if non_conformes - {TEMPORALITE}:
        return StatutClaim.REJECTED_SEMANTIC, motif + " [règle : axe hors temporalité non conforme]"
    if TEMPORALITE in non_conformes:
        if juge is StatutClaim.ESTABLISHED:
            return StatutClaim.CLAIMED_UNPROVEN, motif + " [règle : temporalité non conforme]"
        return juge, motif
    if juge is StatutClaim.CLAIMED_UNPROVEN:
        return StatutClaim.REJECTED_SEMANTIC, motif + " [règle : CLAIMED_UNPROVEN sans axe temporalité]"
    drapeaux = claim.drapeaux
    if drapeaux is not None and drapeaux.is_future_or_protocol and not contient_resultat_chiffre(
        claim.citation_verbatim
    ):
        return StatutClaim.CLAIMED_UNPROVEN, motif + " [règle : futur/protocole sans résultat chiffré]"
    return juge, motif


def statut_preuve(evaluation: EvaluationFenetrePreuve) -> tuple[StatutPreuve, str]:
    axes = {
        "espece_et_modele": evaluation.espece_et_modele,
        TEMPORALITE: evaluation.temporalite_et_statut_de_preuve,
        "mesurande": evaluation.mesurande,
        "population": evaluation.population,
        "intervention": evaluation.intervention,
        "force_de_refutation": evaluation.force_de_refutation,
    }
    statut = StatutPreuve(evaluation.statut)
    motif = _motif(evaluation.indice_contexte, axes, evaluation.motif)
    if statut is StatutPreuve.CONFIRMED and any(a.verdict != "conforme" for a in axes.values()):
        return StatutPreuve.REJECTED_SEMANTIC, motif + " [règle : axe non conforme]"
    return statut, motif


# --------------------------------------------------------------------------- #
# Appels
# --------------------------------------------------------------------------- #


def messages_claim(claim: Claim, question: QuestionD0, document: Document) -> tuple[tuple[Message, ...], int]:
    """Messages exacts envoyés pour un claim, et nombre de contextes (sert aussi à l'estimation)."""
    liste, resume = contextes(claim, document)
    messages = (
        Message(role="system", content=SYSTEME_CLAIM),
        Message(role="user", content=_entree(question, claim, liste, resume, {})),
    )
    return messages, len(liste)


def messages_preuve(
    preuve: PreuveAdverse, cible: Claim, question: QuestionD0, document: Document
) -> tuple[tuple[Message, ...], int]:
    liste, resume = contextes(preuve, document)
    extra: dict[str, object] = {
        "claim_etabli": {"source_id": cible.source_id, "citation_verbatim": cible.citation_verbatim},
        "requete_adverse": preuve.requete,
    }
    messages = (
        Message(role="system", content=SYSTEME_PREUVE),
        Message(role="user", content=_entree(question, preuve, liste, resume, extra)),
    )
    return messages, len(liste)


def auditer_claim(
    claim: Claim, question: QuestionD0, document: Document, client: ClientLLM, ledger: RunLedger
) -> Claim:
    messages, n = messages_claim(claim, question, document)
    liste = range(n)
    try:
        sortie = appeler_structure(client, ledger, RoleLLM.AUDITEUR, messages, SortieAuditeur, _controle(len(liste)))
    except SortieLLMInvalideError as echec:
        return claim.transitionner(StatutClaim.REJECTED_SEMANTIC, Porte.AUDITEUR, echec.motif)
    statut, motif = min((statut_claim(e, claim) for e in sortie.evaluations), key=lambda sm: _RANG_CLAIM[sm[0]])
    return claim.transitionner(statut, Porte.AUDITEUR, motif)


def auditer_preuve(
    preuve: PreuveAdverse,
    cible: Claim,
    question: QuestionD0,
    document: Document,
    client: ClientLLM,
    ledger: RunLedger,
) -> PreuveAdverse:
    messages, n = messages_preuve(preuve, cible, question, document)
    liste = range(n)
    try:
        sortie = appeler_structure(
            client, ledger, RoleLLM.AUDITEUR, messages, SortieAuditeurPreuve, _controle(len(liste))
        )
    except SortieLLMInvalideError as echec:
        return preuve.transitionner(StatutPreuve.REJECTED_SEMANTIC, Porte.AUDITEUR, echec.motif)
    evaluations = [statut_preuve(e) for e in sortie.evaluations]
    rejet = next((sm for sm in evaluations if sm[0] is StatutPreuve.REJECTED_SEMANTIC), None)
    statut, motif = rejet if rejet is not None else evaluations[0]
    return preuve.transitionner(statut, Porte.AUDITEUR, motif)
