"""API HTTP FastAPI pour Mêtis D0 (portes déterministes, démo rejouable et évaluations).

Permet d'exposer les garde-fous hors LLM (vérification verbatim + pré-filtre) sur n'importe
quel texte, de rejouer un cycle D0 complet hors réseau (0 USD, cassettes figées) et de
consulter les rapports d'évaluation mesurés.
"""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import threading
from collections import Counter
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from metis import config
from metis.config import ModeCassette
from metis.literature.arxiv import CollecteurArxiv
from metis.literature.prefilter import prefiltrer_claim
from metis.literature.pubmed import CollecteurPubMed
from metis.literature.verify import verifier_claim
from metis.llm import ClientLLM, RunLedger
from metis.pipeline import cycle_d0
from metis.schemas import (
    BaseLitterature,
    Claim,
    Document,
    Drapeaux,
    EspeceCible,
    FormatSource,
    Grandeur,
    QuestionD0,
    RequetesInitiales,
    StatutClaim,
    VerdictRoutage,
)

app = FastAPI(
    title="Mêtis D0 — Audit documentaire fail-closed",
    version="2.2.0",
    description=(
        "Orchestrateur multi-agents à garde-fous déterministes : citations vérifiées mot pour mot "
        "hors LLM, pré-filtres lexicaux, schémas Pydantic stricts et démo complète rejouable sans clé API."
    ),
)

_VERROU_DEMO = threading.Lock()

_QUESTION_DEMO = QuestionD0(
    question_id="vitd-fractures-cycle",
    sous_question=(
        "Chez l'adulte de 65 ans et plus, une supplémentation en vitamine D3 associée au calcium, contre placebo, "
        "réduit-elle l'incidence des fractures ?"
    ),
    espece_cible=EspeceCible.HUMAN_IN_VIVO,
    population="adultes de 65 ans et plus, hommes et femmes, vivant à domicile ou en institution",
    intervention=(
        "vitamine D3 orale quotidienne (environ 700 à 800 UI/j) associée au calcium (0,5 à 1,2 g/j), "
        "contre placebo"
    ),
    mesurande=Grandeur(
        nom="incidence des fractures (hanche, non vertébrales, toutes fractures)",
        unite="rapport de risques",
    ),
    exclusions=("ostéoporose secondaire", "enfants et adolescents"),
    requetes_initiales=RequetesInitiales(
        pubmed=("vitamin D3 calcium hip fractures elderly women randomized",)
    ),
)


class _TransportRejeuHTTP(httpx.BaseTransport):
    """Transport httpx lecture seule rejouant des réponses HTTP figées (aucun accès réseau)."""

    def __init__(self, dossier: Path) -> None:
        self._dossier = dossier

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        cle = hashlib.sha256(f"{request.method} {request.url}".encode()).hexdigest()
        chemin = self._dossier / f"{cle}.json"
        if not chemin.exists():
            raise FileNotFoundError(f"réponse HTTP figée absente : {request.method} {request.url}")
        donnees: dict[str, object] = json.loads(chemin.read_text(encoding="utf-8"))
        statut = donnees["status"]
        corps_b64 = donnees["corps_b64"]
        if not isinstance(statut, int) or not isinstance(corps_b64, str):
            raise ValueError(f"fixture HTTP invalide : {chemin.name}")
        entetes = {"content-type": str(donnees["content_type"])}
        if donnees.get("location"):
            entetes["location"] = str(donnees["location"])
        return httpx.Response(
            statut,
            headers=entetes,
            content=base64.b64decode(corps_b64),
            request=request,
        )


def _trouver_dossier(relatif: Path) -> Path | None:
    for base in (config.RACINE_DEPOT, Path.cwd()):
        candidat = base / relatif
        if candidat.exists():
            return candidat
    return None


# --------------------------------------------------------------------------- #
# Schémas HTTP
# --------------------------------------------------------------------------- #


class ReponseSante(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    version: str
    mode_cassette: str
    modele_juge: str
    llm_base_url: str
    lexiques_version: str
    textclean_version: str


class RequeteVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(default="doc:1", min_length=1)
    source_text: str = Field(..., min_length=1)
    citation: str = Field(..., min_length=1)
    espece_cible: EspeceCible = EspeceCible.HUMAN_IN_VIVO


class ReponseVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    statut: StatutClaim
    verbatim_ok: bool
    occurrences: int
    motif_rejet: str | None
    drapeaux: Drapeaux | None
    fenetres: tuple[str, ...]


class ReponseAuditDemo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    mode: str
    cout_reel_usd: float
    cout_enregistre_usd: float
    nb_appels_llm_rejoues: int
    nb_documents: int
    verdict: VerdictRoutage | None
    claims_par_statut: dict[str, int]
    rapport_markdown: str


class ResumeJugeEval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    modele: str
    date: str
    nb_cas_total: int
    nb_cas_evalues: int
    nb_cas_mesures: int
    cas_valides_par_humain: int
    exactitude_globale: float | None
    exactitude_hors_refus: float | None
    taux_faux_established: float | None
    taux_faux_rejets: float | None
    taux_faux_rejets_prefiltre: float | None
    nb_cas_refuses_fournisseur: int
    cout_total_usd: float
    nb_appels_llm: int
    interruption_budget: str | None


class ReponseEvals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rapports: tuple[ResumeJugeEval, ...]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=ReponseSante)
def health() -> ReponseSante:
    return ReponseSante(
        status="ok",
        version=app.version,
        mode_cassette=config.mode_cassette().value,
        modele_juge=config.modele_juge(),
        llm_base_url=config.llm_base_url(),
        lexiques_version=config.LEXIQUES_VERSION,
        textclean_version=config.TEXTCLEAN_VERSION,
    )


@app.post("/v1/verify", response_model=ReponseVerification)
def verify_claim_endpoint(requete: RequeteVerification) -> ReponseVerification:
    """Exécute les portes déterministes (verify.py + prefilter.py) sans aucun appel LLM."""
    document = Document(
        source_id=requete.source_id,
        format=FormatSource.PUBMED_XML,
        texte_brut=requete.source_text,
        texte_nettoye=requete.source_text,
        textclean_version=config.TEXTCLEAN_VERSION,
    )
    claim, _ = verifier_claim(
        Claim.extrait("c001", requete.source_id, requete.citation),
        {document.source_id: document},
    )
    verbatim_ok = claim.statut is StatutClaim.EXTRACTED
    if verbatim_ok:
        question = QuestionD0(
            question_id="verification-unitaire",
            sous_question="Vérification unitaire des portes déterministes.",
            espece_cible=requete.espece_cible,
            population="non spécifiée",
            intervention="non spécifiée",
            mesurande=Grandeur(nom="non spécifié", unite="n/a"),
            requetes_initiales=RequetesInitiales(pubmed=("placeholder",)),
        )
        claim = prefiltrer_claim(claim, question)
    return ReponseVerification(
        statut=claim.statut,
        verbatim_ok=verbatim_ok,
        occurrences=claim.nombre_occurrences or 0,
        motif_rejet=claim.motif_rejet,
        drapeaux=claim.drapeaux,
        fenetres=claim.fenetres,
    )


@app.post("/v1/audit/demo", response_model=ReponseAuditDemo)
def audit_demo_endpoint() -> ReponseAuditDemo:
    """Rejoue un cycle D0 complet hors réseau (0 USD) sur les fixtures PubMed et cassettes LLM."""
    dossier_http = _trouver_dossier(Path("tests/fixtures/http/cycle_vitamine_d"))
    dossier_cassettes = _trouver_dossier(Path("tests/fixtures/cassettes"))
    if dossier_http is None or dossier_cassettes is None:
        raise HTTPException(status_code=503, detail="Fixtures de démo introuvables sur le disque.")

    with _VERROU_DEMO, tempfile.TemporaryDirectory(prefix="metis-demo-") as tmp:
        dossier_run = Path(tmp) / "cycle-d0-demo"
        ancien_init = config.MAX_RESULTATS_REQUETE_INITIALE
        ancien_adv = config.MAX_RESULTATS_REQUETE_ADVERSE
        config.MAX_RESULTATS_REQUETE_INITIALE = 3
        config.MAX_RESULTATS_REQUETE_ADVERSE = 2
        try:
            http_client = httpx.Client(transport=_TransportRejeuHTTP(dossier_http), timeout=30.0)
            try:
                ledger = RunLedger("cycle-d0-demo", dossier_run)
                deps = cycle_d0.Dependances(
                    client=ClientLLM(ModeCassette.REPLAY, dossier_cassettes),
                    ledger=ledger,
                    collecteurs={
                        BaseLitterature.PUBMED: CollecteurPubMed(http_client, pause_s=0.0),
                        BaseLitterature.ARXIV: CollecteurArxiv(http_client, pause_s=0.0),
                    },
                    dossier_run=dossier_run,
                )
                etat = cycle_d0.executer_d0(_QUESTION_DEMO, "cycle-d0-demo", deps)
            finally:
                http_client.close()
        finally:
            config.MAX_RESULTATS_REQUETE_INITIALE = ancien_init
            config.MAX_RESULTATS_REQUETE_ADVERSE = ancien_adv

        rapport_md = (dossier_run / "rapport.md").read_text(encoding="utf-8")
        inst = ledger.instantane()
        compteurs = Counter(c.statut.value for c in etat.claims)
        return ReponseAuditDemo(
            run_id=etat.run_id,
            mode=ModeCassette.REPLAY.value,
            cout_reel_usd=ledger.depense_reelle_usd(),
            cout_enregistre_usd=round(inst.cout_total_usd, 6),
            nb_appels_llm_rejoues=len(inst.entrees),
            nb_documents=len(etat.documents),
            verdict=etat.verdict,
            claims_par_statut=dict(sorted(compteurs.items())),
            rapport_markdown=rapport_md,
        )


def _float_ou_none(valeur: object) -> float | None:
    return float(valeur) if isinstance(valeur, (int, float)) else None


def _int_ou_zero(valeur: object) -> int:
    return int(valeur) if isinstance(valeur, int) else 0


@app.get("/v1/evals", response_model=ReponseEvals)
def evals_endpoint() -> ReponseEvals:
    """Expose la synthèse des rapports d'évaluation mesurés sur le gold set D0."""
    dossier_evals = _trouver_dossier(Path("evals"))
    if dossier_evals is None:
        raise HTTPException(status_code=503, detail="Dossier evals/ introuvable.")

    resumes: list[ResumeJugeEval] = []
    for chemin in sorted(dossier_evals.glob("rapport_*.json")):
        donnees: dict[str, object] = json.loads(chemin.read_text(encoding="utf-8"))
        par_juge = donnees.get("par_juge")
        if not isinstance(par_juge, dict):
            continue
        date = str(donnees.get("date", ""))
        nb_cas = _int_ou_zero(donnees.get("nb_cas"))
        valides = _int_ou_zero(donnees.get("cas_valides_par_humain"))
        for modele, stats in par_juge.items():
            if not isinstance(modele, str) or not isinstance(stats, dict):
                continue
            interruption = stats.get("interruption_budget")
            resumes.append(
                ResumeJugeEval(
                    modele=modele,
                    date=date,
                    nb_cas_total=nb_cas,
                    nb_cas_evalues=_int_ou_zero(stats.get("nb_cas_evalues")),
                    nb_cas_mesures=_int_ou_zero(stats.get("nb_cas_mesures")),
                    cas_valides_par_humain=valides,
                    exactitude_globale=_float_ou_none(stats.get("exactitude_globale")),
                    exactitude_hors_refus=_float_ou_none(stats.get("exactitude_hors_refus")),
                    taux_faux_established=_float_ou_none(stats.get("taux_faux_established")),
                    taux_faux_rejets=_float_ou_none(stats.get("taux_faux_rejets")),
                    taux_faux_rejets_prefiltre=_float_ou_none(stats.get("taux_faux_rejets_prefiltre")),
                    nb_cas_refuses_fournisseur=_int_ou_zero(stats.get("nb_cas_refuses")),
                    cout_total_usd=_float_ou_none(stats.get("cout_total_usd")) or 0.0,
                    nb_appels_llm=_int_ou_zero(stats.get("nb_appels_llm")),
                    interruption_budget=str(interruption) if interruption is not None else None,
                )
            )
    return ReponseEvals(rapports=tuple(resumes))
