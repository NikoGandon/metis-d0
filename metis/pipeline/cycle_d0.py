"""Orchestration D0 : chaîne explicite d'étapes EtatD0 -> EtatD0, persistée après chacune.

collecter -> extraire -> filtrer (verify, prefilter) -> auditer -> chercher_refutations
-> evaluer_preuves -> router -> rapport. Le flux de contrôle est déterministe ; les
sorties LLM sont enregistrées (cassettes, reponses_llm/).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from metis import config
from metis.literature.prefilter import prefiltrer_claim, prefiltrer_preuve
from metis.literature.verify import verifier_claim, verifier_preuve
from metis.llm import ClientLLM, RunLedger, SortieLLMInvalideError
from metis.pipeline import auditeur, contradicteur, extracteur, routeur
from metis.rapport.markdown import ecrire_rapport
from metis.schemas import (
    BaseLitterature,
    Claim,
    Document,
    EtatD0,
    Incident,
    OrigineRequete,
    Porte,
    PreuveAdverse,
    QuestionD0,
    RequeteJournal,
    StatutClaim,
    StatutPreuve,
)
from metis.storage import ecrire_json_atomique


class Collecteur(Protocol):
    base: BaseLitterature

    def collecter(self, requete: str, max_resultats: int) -> tuple[Document, ...]: ...


@dataclass(frozen=True)
class Dependances:
    client: ClientLLM
    ledger: RunLedger
    collecteurs: Mapping[BaseLitterature, Collecteur]
    dossier_run: Path


# --------------------------------------------------------------------------- #
# Persistance
# --------------------------------------------------------------------------- #


def nom_fichier_document(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", source_id) + ".json"


def persister(etat: EtatD0, dossier: Path) -> None:
    ecrire_json_atomique(dossier / "question.json", etat.question)
    noms: dict[str, str] = {}
    for document in etat.documents:
        nom = nom_fichier_document(document.source_id)
        if noms.setdefault(nom, document.source_id) != document.source_id:
            raise ValueError(f"collision de noms de fichier : {nom}")
        chemin = dossier / "corpus" / nom
        if not chemin.exists():
            ecrire_json_atomique(chemin, document)
    ecrire_json_atomique(dossier / "requetes.json", [r.model_dump(mode="json") for r in etat.requetes])
    ecrire_json_atomique(dossier / "claims.json", [c.model_dump(mode="json") for c in etat.claims])
    ecrire_json_atomique(dossier / "preuves_adverses.json", [p.model_dump(mode="json") for p in etat.preuves])
    ecrire_json_atomique(
        dossier / "candidats_equation.json", [c.model_dump(mode="json") for c in etat.candidats_equation]
    )
    ecrire_json_atomique(dossier / "incidents.json", [i.model_dump(mode="json") for i in etat.incidents])
    if etat.verdict is not None:
        ecrire_json_atomique(dossier / "verdict.json", etat.verdict)


def _maj(etat: EtatD0, deps: Dependances, **champs: object) -> EtatD0:
    nouvel = EtatD0.model_validate({**dict(etat), **champs})
    persister(nouvel, deps.dossier_run)
    return nouvel


# --------------------------------------------------------------------------- #
# Étapes
# --------------------------------------------------------------------------- #


def _executer_requete(
    etat: EtatD0,
    deps: Dependances,
    base: BaseLitterature,
    texte: str,
    origine: OrigineRequete,
    claim_cible_id: str | None,
    max_resultats: int,
) -> tuple[EtatD0, tuple[str, ...]]:
    collecteur = deps.collecteurs.get(base)
    erreur: str | None = None
    documents: tuple[Document, ...] = ()
    if collecteur is None:
        erreur = f"aucun collecteur pour {base}"
    else:
        try:
            documents = collecteur.collecter(texte, max_resultats)
        except (httpx.HTTPError, ET.ParseError, ValueError, KeyError) as exc:
            erreur = f"{type(exc).__name__}: {exc}"
    deja = {d.source_id for d in etat.documents}
    nouveaux = tuple(d for d in documents if d.source_id not in deja)
    journal = RequeteJournal(
        origine=origine,
        base=base,
        texte=texte,
        claim_cible_id=claim_cible_id,
        source_ids=tuple(d.source_id for d in documents),
        erreur=erreur,
    )
    etat = _maj(etat, deps, documents=(*etat.documents, *nouveaux), requetes=(*etat.requetes, journal))
    return etat, journal.source_ids


def collecter(etat: EtatD0, deps: Dependances) -> EtatD0:
    initiales = etat.question.requetes_initiales
    for base, requetes in ((BaseLitterature.PUBMED, initiales.pubmed), (BaseLitterature.ARXIV, initiales.arxiv)):
        for texte in requetes:
            etat, _ = _executer_requete(
                etat, deps, base, texte, OrigineRequete.INITIALE, None, config.MAX_RESULTATS_REQUETE_INITIALE
            )
    return etat


def extraire(etat: EtatD0, deps: Dependances) -> EtatD0:
    claims, equations, incidents = extracteur.extraire(
        etat.question, etat.documents, deps.client, deps.ledger, numero_initial=len(etat.claims) + 1
    )
    return _maj(
        etat,
        deps,
        claims=(*etat.claims, *claims),
        candidats_equation=(*etat.candidats_equation, *equations),
        incidents=(*etat.incidents, *incidents),
    )


def _a_filtrer(claim: Claim) -> bool:
    return claim.statut is StatutClaim.EXTRACTED and claim.drapeaux is None


def filtrer(etat: EtatD0, deps: Dependances) -> EtatD0:
    """verify.py puis prefilter.py (0 LLM) sur les claims EXTRACTED non encore filtrés."""
    corpus = etat.corpus
    claims = []
    for claim in etat.claims:
        if _a_filtrer(claim):
            claim, _ = verifier_claim(claim, corpus)
            if claim.statut is StatutClaim.EXTRACTED:
                claim = prefiltrer_claim(claim, etat.question)
        claims.append(claim)
    return _maj(etat, deps, claims=tuple(claims))


def auditer(etat: EtatD0, deps: Dependances) -> EtatD0:
    corpus = etat.corpus
    claims = tuple(
        auditeur.auditer_claim(c, etat.question, corpus[c.source_id], deps.client, deps.ledger)
        if c.statut is StatutClaim.EXTRACTED and c.drapeaux is not None
        else c
        for c in etat.claims
    )
    return _maj(etat, deps, claims=claims)


def _incident(etat: EtatD0, deps: Dependances, claim_id: str, motif: str) -> EtatD0:
    """Incident du contradicteur : le claim ne pourra pas fonder un verdict simulable."""
    incident = Incident(etape="contradicteur", cible=claim_id, motif=motif)
    return _maj(etat, deps, incidents=(*etat.incidents, incident))


def chercher_refutations(etat: EtatD0, deps: Dependances) -> EtatD0:
    """Requêtes adverses + propositions -> PreuveAdverse CANDIDATE (non encore évaluées)."""
    for claim in [c for c in etat.claims if c.statut is StatutClaim.ESTABLISHED]:
        try:
            requetes = contradicteur.formuler_requetes(claim, etat.question, deps.client, deps.ledger)
        except SortieLLMInvalideError as echec:
            etat = _incident(etat, deps, claim.claim_id, f"{echec.motif} (requêtes)")
            continue
        source_ids: list[str] = []
        for requete in requetes:
            etat, trouves = _executer_requete(
                etat,
                deps,
                BaseLitterature(requete.base),
                requete.texte,
                OrigineRequete.ADVERSE,
                claim.claim_id,
                config.MAX_RESULTATS_REQUETE_ADVERSE,
            )
            source_ids.extend(s for s in trouves if s not in source_ids)
        corpus = etat.corpus
        preuves = list(etat.preuves)
        for source_id in source_ids:
            try:
                proposees = contradicteur.proposer_preuves(
                    claim, etat.question, corpus[source_id], deps.client, deps.ledger
                )
            except SortieLLMInvalideError as echec:
                etat = _incident(etat, deps, claim.claim_id, f"{echec.motif} ({source_id})")
                continue
            requete_origine = next(
                r.texte
                for r in reversed(etat.requetes)
                if r.claim_cible_id == claim.claim_id and source_id in r.source_ids
            )
            for p in proposees:
                identifiant = f"p{len(preuves) + 1:03d}"
                preuves.append(
                    PreuveAdverse.candidate(identifiant, claim.claim_id, requete_origine, p.source_id, p.citation_verbatim)
                )
        etat = _maj(etat, deps, preuves=tuple(preuves))
    return etat


def evaluer_preuves(etat: EtatD0, deps: Dependances) -> EtatD0:
    """Chaque PreuveAdverse CANDIDATE passe verify.py, prefilter.py puis l'auditeur ;
    une preuve CONFIRMED fait passer son claim cible en CONTRADICTED."""
    corpus = etat.corpus
    par_id = {c.claim_id: c for c in etat.claims}
    preuves = []
    for preuve in etat.preuves:
        if preuve.statut_preuve is StatutPreuve.CANDIDATE and preuve.drapeaux is None:
            preuve, _ = verifier_preuve(preuve, corpus)
            if preuve.statut_preuve is StatutPreuve.CANDIDATE:
                preuve = prefiltrer_preuve(preuve, etat.question)
            if preuve.statut_preuve is StatutPreuve.CANDIDATE:
                preuve = auditeur.auditer_preuve(
                    preuve, par_id[preuve.claim_cible_id], etat.question, corpus[preuve.source_id],
                    deps.client, deps.ledger,
                )
        preuves.append(preuve)
        cible = par_id[preuve.claim_cible_id]
        if preuve.statut_preuve is StatutPreuve.CONFIRMED and cible.statut is StatutClaim.ESTABLISHED:
            par_id[cible.claim_id] = cible.transitionner(StatutClaim.CONTRADICTED, Porte.CONTRADICTEUR, preuve.preuve_id)
    claims = tuple(par_id[c.claim_id] for c in etat.claims)
    return _maj(etat, deps, claims=claims, preuves=tuple(preuves))


def router(etat: EtatD0, deps: Dependances) -> EtatD0:
    verdict = routeur.router(
        etat.question, etat.claims, etat.incidents, etat.candidats_equation, etat.corpus, deps.client, deps.ledger
    )
    return _maj(etat, deps, verdict=verdict)


# --------------------------------------------------------------------------- #
# Run complet
# --------------------------------------------------------------------------- #


def demarrer(question: QuestionD0, run_id: str, deps: Dependances, documents: Sequence[Document] = ()) -> EtatD0:
    return _maj(EtatD0(run_id=run_id, question=question), deps, documents=tuple(documents))


def executer_d0(question: QuestionD0, run_id: str, deps: Dependances) -> EtatD0:
    etat = demarrer(question, run_id, deps)
    for etape in (collecter, extraire, filtrer, auditer, chercher_refutations, evaluer_preuves, router):
        etat = etape(etat, deps)
    ecrire_rapport(deps.dossier_run)
    return etat
