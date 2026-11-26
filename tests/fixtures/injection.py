"""Aide aux tests et aux évaluations : injection d'un claim en ESTABLISHED sans l'auditeur.

Volontairement hors de metis/ : aucun code du paquet ne peut faire passer un claim en
ESTABLISHED sans l'auditeur (vérifié par test_cycle_d0.py). Sert uniquement à éprouver
le chemin de réfutation, indépendamment du sort du claim cible.
"""

from __future__ import annotations

from metis.schemas import Claim, Porte, StatutClaim

MOTIF_INJECTION = "injecté pour tester le chemin de réfutation"


def injecter_etabli(claim: Claim) -> Claim:
    """Porte auditeur simulée : le claim (gates passées) passe en ESTABLISHED sans appel au juge."""
    return claim.transitionner(StatutClaim.ESTABLISHED, Porte.AUDITEUR, MOTIF_INJECTION)
