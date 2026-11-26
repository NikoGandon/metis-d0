"""Gate prefilter.py : lexiques à limites de mot et drapeaux (0 LLM).

Les lexiques (config.py, versionnés) sont appliqués au texte normalisé par verify.py.
Portée des drapeaux :
- species_hits et human_hits : la citation elle-même (seule base de la règle de rejet) ;
- is_future_or_protocol : toutes les fenêtres de contexte conservées, plus larges,
  car ce drapeau ne fait qu'alerter l'auditeur (il ne change jamais le statut ici).
Un marqueur humain suivi d'un trait d'union ou de « derived » n'est pas compté.

Seule transition autorisée : REJECTED_PREFILTER quand espece_cible == human_in_vivo,
qu'une espèce ou un système non in vivo est cité et qu'aucun marqueur humain ne
figure dans la citation.
"""

from __future__ import annotations

import re

from metis import config
from metis.literature.verify import normaliser
from metis.schemas import (
    Claim,
    Drapeaux,
    EspeceCible,
    Porte,
    PreuveAdverse,
    QuestionD0,
    StatutClaim,
    StatutPreuve,
)


# Un marqueur humain collé à un trait d'union ou suivi de « derived » qualifie un objet
# non humain (patient-derived xenografts, human-derived cells) : il n'est pas compté.
_NON_DERIVE = r"(?!-|\s+derived\b)"


def _compiler(motifs: tuple[str, ...], suffixe: str = "") -> tuple[re.Pattern[str], ...]:
    for motif in motifs:
        if not (motif.startswith(r"\b") and motif.endswith(r"\b")):
            raise ValueError(f"expression sans limites de mot : {motif!r}")
    return tuple(re.compile(motif + suffixe, re.IGNORECASE) for motif in motifs)


_ESPECES = _compiler(config.LEXIQUE_ESPECES_NON_HUMAINES + config.LEXIQUE_SYSTEMES_NON_IN_VIVO)
_HUMAINS = _compiler(config.LEXIQUE_MARQUEURS_HUMAINS, _NON_DERIVE)
_FUTUR = _compiler(config.LEXIQUE_FUTUR_PROTOCOLE)


def _hits(motifs: tuple[re.Pattern[str], ...], texte: str) -> tuple[str, ...]:
    """Occurrences distinctes, dans l'ordre d'apparition."""
    trouvees = sorted((m.start(), m.group(0)) for motif in motifs for m in motif.finditer(texte))
    return tuple(dict.fromkeys(t for _, t in trouvees))


def calculer_drapeaux(citation: str, fenetres: tuple[str, ...]) -> Drapeaux:
    citation_n = normaliser(citation).texte
    return Drapeaux(
        is_future_or_protocol=any(_hits(_FUTUR, normaliser(f).texte) for f in fenetres),
        species_hits=_hits(_ESPECES, citation_n),
        human_hits=_hits(_HUMAINS, citation_n),
    )


def motif_rejet(drapeaux: Drapeaux, espece_cible: EspeceCible) -> str | None:
    if espece_cible is EspeceCible.HUMAN_IN_VIVO and drapeaux.species_hits and not drapeaux.human_hits:
        return "espece_exclue_sans_marqueur_humain: " + ", ".join(drapeaux.species_hits)
    return None


def prefiltrer_claim(claim: Claim, question: QuestionD0) -> Claim:
    if claim.context_window is None:
        raise ValueError("prefilter exige un claim ancré par verify.py")
    drapeaux = calculer_drapeaux(claim.citation_verbatim, claim.fenetres)
    claim = claim.avec_drapeaux(drapeaux)
    motif = motif_rejet(drapeaux, question.espece_cible)
    return claim if motif is None else claim.transitionner(StatutClaim.REJECTED_PREFILTER, Porte.PREFILTER, motif)


def prefiltrer_preuve(preuve: PreuveAdverse, question: QuestionD0) -> PreuveAdverse:
    if preuve.context_window is None:
        raise ValueError("prefilter exige une preuve ancrée par verify.py")
    drapeaux = calculer_drapeaux(preuve.citation_verbatim, preuve.fenetres)
    preuve = preuve.avec_drapeaux(drapeaux)
    motif = motif_rejet(drapeaux, question.espece_cible)
    if motif is None:
        return preuve
    return preuve.transitionner(StatutPreuve.REJECTED_PREFILTER, Porte.PREFILTER, motif)
