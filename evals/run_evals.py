"""Mesure et comparaison des juges D0 sur le gold set (appels LLM réels, hors pytest).

Usage :
    OPENROUTER_API_KEY=... .venv/bin/python evals/run_evals.py [--juges m1,m2,...]
    .venv/bin/python evals/run_evals.py --recalculer [--juges m1,m2,...]

Par défaut, compare les juges de config.JUGES_COMPARES. Chaque réponse est enregistrée
en cassette dans evals/cassettes/ (mode record_missing : une réponse déjà enregistrée
n'est jamais redemandée). --recalculer rejoue ces cassettes sans aucun appel réseau
et recalcule les métriques. Chaque juge a son propre RunLedger et son propre budget :
METIS_BUDGET_USD, 1 USD par juge par défaut pour les évaluations.

Chaque cas passe par les mêmes portes que dans un run : verify.py, prefilter.py puis
l'auditeur (claims) ; pour les cas de réfutation, la preuve adverse passe verify.py,
prefilter.py puis l'auditeur avec l'axe « force de réfutation ». Les notices viennent
des réponses PubMed figées (tests/fixtures/http/gold/).

Produit evals/rapport_<date>.json. Métrique de premier rang : le taux de
refus_fournisseur par modèle et par rôle, séparément sur les cas à sujet sensible
(vaccins, immunosuppression, infectiologie) et les autres. Puis, par juge : taux de
faux ESTABLISHED, de faux rejets (dont ceux du pré-filtre), de fausses CONFIRMED du
contradicteur, coût. Seuil d'alerte : faux ESTABLISHED < 5 %. Ce rapport n'est pas
un test pytest. Un refus reste fail-closed : le cas compte comme jugé REJECTED_SEMANTIC.
Les cas étiquetés AMBIGU sont exclus des métriques et listés à part.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "tests" / "fixtures"))
sys.path.insert(0, str(RACINE))

from http_fige import client_fige  # noqa: E402
from injection import injecter_etabli  # noqa: E402

from metis import config  # noqa: E402
from metis.literature.prefilter import prefiltrer_claim, prefiltrer_preuve  # noqa: E402
from metis.literature.pubmed import CollecteurPubMed  # noqa: E402
from metis.literature.verify import verifier_claim, verifier_preuve  # noqa: E402
from metis.config import ModeCassette  # noqa: E402
from metis.llm import (  # noqa: E402
    BudgetDepasseError,
    CassetteMissError,
    ClientLLM,
    RunLedger,
    cle_cassette,
    cout_maximal_usd,
    cout_usd,
    tarif,
)
from metis.pipeline.auditeur import auditer_claim, auditer_preuve, messages_claim, messages_preuve  # noqa: E402
from metis.llm import chemin_cassette  # noqa: E402
from metis.schemas import (  # noqa: E402
    Claim,
    CoutsRun,
    RoleLLM,
    SortieAuditeur,
    SortieAuditeurPreuve,
    Document,
    PreuveAdverse,
    QuestionD0,
    StatutClaim,
    StatutPreuve,
)

GOLD = Path(__file__).parent / "gold_d0.jsonl"
CASSETTES_EVALS = Path(__file__).parent / "cassettes"
BUDGET_PAR_JUGE_DEFAUT = 1.0
HTTP_GOLD = RACINE / "tests" / "fixtures" / "http" / "gold"
SEUIL_FAUX_ETABLIS = 0.05


class _Gele(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Etiquetage(_Gele):
    auteur: str
    date: str
    valide_par_humain: bool
    note: str


class Citation(_Gele):
    source_id: str
    citation: str
    requete: str | None = None


class CasClaim(_Gele):
    cas_id: str
    type: Literal["claim"]
    categorie: str
    question: QuestionD0
    source_id: str
    citation: str
    statuts_acceptables: tuple[StatutClaim, ...]
    sujet_sensible: bool = False
    ambigu: bool = False  # étiquette AMBIGU : exclu des métriques, listé à part
    etiquetage: Etiquetage


class CasRefutation(_Gele):
    cas_id: str
    type: Literal["refutation"]
    categorie: str
    question: QuestionD0
    claim: Citation
    preuve: Citation
    statuts_acceptables: tuple[StatutPreuve, ...]
    sujet_sensible: bool = False
    ambigu: bool = False  # étiquette AMBIGU : exclu des métriques, listé à part
    etiquetage: Etiquetage


def charger_gold(chemin: Path = GOLD) -> list[CasClaim | CasRefutation]:
    cas: list[CasClaim | CasRefutation] = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        donnees = json.loads(ligne)
        modele = CasClaim if donnees["type"] == "claim" else CasRefutation
        cas.append(modele.model_validate(donnees))
    return cas


def corpus_gold(cas: list[CasClaim | CasRefutation], enregistrer: bool = False) -> dict[str, Document]:
    source_ids = set()
    for c in cas:
        source_ids |= {c.source_id} if isinstance(c, CasClaim) else {c.claim.source_id, c.preuve.source_id}
    collecteur = CollecteurPubMed(client_fige(HTTP_GOLD, enregistrer), pause_s=config.PAUSE_NCBI_S if enregistrer else 0)
    return {s: collecteur.notice(s.removeprefix("pmid:"))[0] for s in sorted(source_ids)}


def _preparer_claim(cas: CasClaim, corpus: Mapping[str, Document]) -> Claim:
    """Gates déterministes (verify.py, prefilter.py), sans appel."""
    claim, _ = verifier_claim(Claim.extrait("c001", cas.source_id, cas.citation), corpus)
    return prefiltrer_claim(claim, cas.question) if claim.statut is StatutClaim.EXTRACTED else claim


def _preparer_refutation(cas: CasRefutation, corpus: Mapping[str, Document]) -> tuple[Claim, PreuveAdverse]:
    # Le claim cible est injecté en ESTABLISHED (porte auditeur simulée, sans appel) : le
    # cas mesure le jugement de la preuve, indépendamment du sort du claim lui-même.
    cible, _ = verifier_claim(Claim.extrait("c001", cas.claim.source_id, cas.claim.citation), corpus)
    if cible.statut is not StatutClaim.EXTRACTED:
        raise ValueError(f"{cas.cas_id} : claim cible non verbatim ({cible.motif_rejet})")
    cible = prefiltrer_claim(cible, cas.question)
    if cible.statut is StatutClaim.EXTRACTED:
        cible = injecter_etabli(cible)
    preuve = PreuveAdverse.candidate(
        "p001", "c001", cas.preuve.requete or "gold", cas.preuve.source_id, cas.preuve.citation
    )
    preuve, _ = verifier_preuve(preuve, corpus)
    if preuve.statut_preuve is StatutPreuve.CANDIDATE:
        preuve = prefiltrer_preuve(preuve, cas.question)
    return cible, preuve


def _evaluer_claim(
    cas: CasClaim, corpus: Mapping[str, Document], client: ClientLLM, ledger: RunLedger
) -> tuple[StatutClaim, str | None]:
    claim = _preparer_claim(cas, corpus)
    if claim.statut is StatutClaim.EXTRACTED:
        claim = auditer_claim(claim, cas.question, corpus[claim.source_id], client, ledger)
    return claim.statut, claim.historique[-1].motif


def _evaluer_refutation(
    cas: CasRefutation, corpus: Mapping[str, Document], client: ClientLLM, ledger: RunLedger
) -> tuple[StatutPreuve, str | None]:
    cible, preuve = _preparer_refutation(cas, corpus)
    if preuve.statut_preuve is StatutPreuve.CANDIDATE:
        preuve = auditer_preuve(preuve, cible, cas.question, corpus[preuve.source_id], client, ledger)
    return preuve.statut_preuve, preuve.historique[-1].motif


class ResultatCas(_Gele):
    cas_id: str
    type: Literal["claim", "refutation"]
    categorie: str
    sujet_sensible: bool
    ambigu: bool
    attendus: tuple[str, ...]
    obtenu: str
    motif: str | None

    @property
    def refus_fournisseur(self) -> bool:
        return self.motif == "refus_fournisseur"

    @property
    def correct(self) -> bool:
        return self.obtenu in self.attendus


class NonEvalue(_Gele):
    """Cas sans cassette lors d'un --recalculer : non évalué, aucun appel réseau tenté."""

    cas_id: str
    categorie: str
    raison: str


def _taux(numerateur: int, denominateur: int) -> float | None:
    return numerateur / denominateur if denominateur else None


def _taux_refus(refus: int, total: int) -> dict[str, object]:
    return {"refus": refus, "total": total, "taux": _taux(refus, total)}


def budget_par_juge() -> float:
    valeur = float(os.environ.get(config.ENV_BUDGET, BUDGET_PAR_JUGE_DEFAUT))
    if not 0 < valeur < float("inf"):
        raise ValueError(f"{config.ENV_BUDGET} doit être un nombre fini > 0")
    return valeur


def evaluer_juge(
    juge: str,
    cas: list[CasClaim | CasRefutation],
    corpus: Mapping[str, Document],
    dossier: Path,
    recalculer: bool = False,
) -> tuple[list[ResultatCas], list[NonEvalue], RunLedger, str | None]:
    """Évalue tous les cas avec ce juge ; s'arrête proprement si le budget est atteint.

    En recalcul (rejeu pur), un cas dont une cassette manque est listé non évalué.
    """
    os.environ[config.ENV_MODELE_JUGE] = juge
    for variable in config.ENV_MODELE_PAR_ROLE.values():
        os.environ.pop(variable, None)  # le juge comparé occupe tous les rôles de juge
    ledger = RunLedger(dossier.name, dossier, budget_usd=budget_par_juge())
    mode = ModeCassette.REPLAY if recalculer else ModeCassette.RECORD_MISSING
    client = ClientLLM(mode, CASSETTES_EVALS)
    resultats: list[ResultatCas] = []
    non_evalues: list[NonEvalue] = []
    for c in cas:
        try:
            if isinstance(c, CasClaim):
                statut, motif = _evaluer_claim(c, corpus, client, ledger)
                obtenu, attendus = statut.value, tuple(s.value for s in c.statuts_acceptables)
            else:
                statut_p, motif = _evaluer_refutation(c, corpus, client, ledger)
                obtenu, attendus = statut_p.value, tuple(s.value for s in c.statuts_acceptables)
        except BudgetDepasseError as exc:
            return resultats, non_evalues, ledger, str(exc)
        except CassetteMissError:
            non_evalues.append(NonEvalue(cas_id=c.cas_id, categorie=c.categorie, raison="cassette absente"))
            print(f"[{juge}] {c.cas_id} {c.categorie:<24} non évalué (cassette absente)")
            continue
        resultat = ResultatCas(
            cas_id=c.cas_id,
            type=c.type,
            categorie=c.categorie,
            sujet_sensible=c.sujet_sensible,
            ambigu=c.ambigu,
            attendus=attendus,
            obtenu=obtenu,
            motif=motif,
        )
        resultats.append(resultat)
        print(f"[{juge}] {c.cas_id} {c.categorie:<24} attendu {'/'.join(attendus):<36} obtenu {obtenu}")
    return resultats, non_evalues, ledger, None


def metriques(tous: Sequence[ResultatCas], ledger: RunLedger) -> dict[str, object]:
    resultats = [r for r in tous if not r.ambigu]  # AMBIGU exclus des métriques
    claims = [r for r in resultats if r.type == "claim"]
    non_etablis = [r for r in claims if "ESTABLISHED" not in r.attendus]
    etablis = [r for r in claims if "ESTABLISHED" in r.attendus]
    refutations = [r for r in resultats if r.type == "refutation"]
    fausses_refutations = [r for r in refutations if "CONFIRMED" not in r.attendus]
    vraies_refutations = [r for r in refutations if "CONFIRMED" in r.attendus]
    couts = ledger.instantane()
    taux_faux_etablis = _taux(sum(r.obtenu == "ESTABLISHED" for r in non_etablis), len(non_etablis))
    return {
        "nb_cas_evalues": len(tous),
        "nb_cas_mesures": len(resultats),
        "cas_ambigus": [
            {"cas_id": r.cas_id, "categorie": r.categorie, "attendus": r.attendus, "obtenu": r.obtenu}
            for r in tous
            if r.ambigu
        ],
        "taux_faux_established": taux_faux_etablis,
        "alerte_faux_established": taux_faux_etablis is not None and taux_faux_etablis >= SEUIL_FAUX_ETABLIS,
        "taux_faux_rejets": _taux(sum(r.obtenu != "ESTABLISHED" for r in etablis), len(etablis)),
        "taux_faux_rejets_prefiltre": _taux(sum(r.obtenu == "REJECTED_PREFILTER" for r in etablis), len(etablis)),
        "taux_fausses_confirmed_contradicteur": _taux(
            sum(r.obtenu == "CONFIRMED" for r in fausses_refutations), len(fausses_refutations)
        ),
        # Alerte si aucune vraie réfutation (non ambiguë) n'est CONFIRMED : le contradicteur
        # serait incapable de faire tomber un claim.
        "alerte_aucune_confirmed": bool(vraies_refutations)
        and not any(r.obtenu == "CONFIRMED" for r in vraies_refutations),
        "taux_vraies_refutations_manquees": _taux(
            sum(r.obtenu != "CONFIRMED" for r in vraies_refutations), len(vraies_refutations)
        ),
        "exactitude_globale": _taux(sum(r.correct for r in resultats), len(resultats)),
        # Un refus du fournisseur finit en REJECTED_SEMANTIC (fail-closed) : il « réussit » les cas
        # attendus REJECTED_SEMANTIC sans que le juge ait jugé. Mesures hors refus :
        "nb_cas_refuses": sum(r.refus_fournisseur for r in resultats),
        "nb_correct_par_refus": sum(r.refus_fournisseur and r.correct for r in resultats),
        "exactitude_hors_refus": _taux(
            sum(r.correct for r in resultats if not r.refus_fournisseur),
            sum(not r.refus_fournisseur for r in resultats),
        ),
        "taux_faux_rejets_hors_refus": _taux(
            sum(r.obtenu != "ESTABLISHED" for r in etablis if not r.refus_fournisseur),
            sum(not r.refus_fournisseur for r in etablis),
        ),
        "cout_total_usd": couts.cout_total_usd,
        "nb_appels_llm": len(couts.entrees),
    }


def refus_par_role(ledger: RunLedger) -> dict[str, dict[str, object]]:
    """Compteur refus_fournisseur par rôle, sur les appels du ledger (retries compris)."""
    compte: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in ledger.entrees:
        compte[e.role.value][0] += e.refus_fournisseur
        compte[e.role.value][1] += 1
    return {role: _taux_refus(*compte[role]) for role in sorted(compte)}


def refus_par_sujet(resultats: Sequence[ResultatCas]) -> dict[str, dict[str, object]]:
    """Part des cas finissant en refus_fournisseur, sujets sensibles d'un côté, autres de l'autre."""
    sortie = {}
    for nom, sensible in (("sujets_sensibles", True), ("autres_sujets", False)):
        groupe = [r for r in resultats if r.sujet_sensible is sensible]
        sortie[nom] = _taux_refus(sum(r.refus_fournisseur for r in groupe), len(groupe))
    return sortie


# --------------------------------------------------------------------------- #
# Estimation du coût (aucun appel)
# --------------------------------------------------------------------------- #

RUNS_EVALS = Path(__file__).parent / "runs"


class AppelPrevu(_Gele):
    cas_id: str
    role: RoleLLM
    deja_enregistre: bool  # cassette présente : rejouée, non facturée
    cout_maximal_usd: float


def appels_prevus(juge: str, cas: Sequence[CasClaim | CasRefutation], corpus: Mapping[str, Document]) -> list[AppelPrevu]:
    """Premier appel de l'auditeur de chaque cas qui franchit les gates (retries non comptés)."""
    t = tarif(juge)
    fournisseurs = config.fournisseurs_pour_role(RoleLLM.AUDITEUR)
    prevus = []
    for c in cas:
        if isinstance(c, CasClaim):
            claim = _preparer_claim(c, corpus)
            if claim.statut is not StatutClaim.EXTRACTED:
                continue
            messages, _ = messages_claim(claim, c.question, corpus[claim.source_id])
            schema: type[BaseModel] = SortieAuditeur
        else:
            cible, preuve = _preparer_refutation(c, corpus)
            if preuve.statut_preuve is not StatutPreuve.CANDIDATE:
                continue
            messages, _ = messages_preuve(preuve, cible, c.question, corpus[preuve.source_id])
            schema = SortieAuditeurPreuve
        cle = cle_cassette(juge, messages, schema, fournisseurs)
        prevus.append(
            AppelPrevu(
                cas_id=c.cas_id,
                role=RoleLLM.AUDITEUR,
                deja_enregistre=chemin_cassette(CASSETTES_EVALS, cle).exists(),
                cout_maximal_usd=cout_maximal_usd(t, messages, schema, config.max_tokens_pour_role(RoleLLM.AUDITEUR)),
            )
        )
    return prevus


def couts_mesures(racine: Path = RUNS_EVALS) -> dict[tuple[str, str], dict[str, float]]:
    """Moyennes des appels réellement facturés (hors rejeux), par (modèle, rôle)."""
    cumul: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for chemin in sorted(racine.glob("*/*/couts.json")):
        for e in CoutsRun.model_validate_json(chemin.read_text(encoding="utf-8")).entrees:
            if e.cassette:
                continue
            ligne = cumul[(e.model, e.role.value)]
            ligne[0] += 1
            ligne[1] += e.prompt_tokens
            ligne[2] += e.completion_tokens
            ligne[3] += e.cost_usd
    return {
        cle: {"appels": n, "tokens_entree": te / n, "tokens_sortie": ts / n, "cout_usd": c / n}
        for cle, (n, te, ts, c) in cumul.items()
    }


def estimer(
    juge: str, prevus: Sequence[AppelPrevu], mesures: Mapping[tuple[str, str], Mapping[str, float]]
) -> dict[str, object]:
    role = RoleLLM.AUDITEUR.value
    restants = [a for a in prevus if not a.deja_enregistre]
    if (juge, role) in mesures:
        moyen, methode = mesures[(juge, role)]["cout_usd"], f"coût moyen mesuré ({juge}, {role})"
    else:
        autres = [m for (modele, r), m in mesures.items() if r == role]
        if autres:
            n = sum(m["appels"] for m in autres)
            entree = sum(m["tokens_entree"] * m["appels"] for m in autres) / n
            sortie = sum(m["tokens_sortie"] * m["appels"] for m in autres) / n
            moyen = cout_usd(tarif(juge), round(entree), round(sortie))
            methode = f"tokens moyens mesurés sur les autres modèles × tarif de {juge}"
        else:
            moyen, methode = max((a.cout_maximal_usd for a in prevus), default=0.0), "aucune mesure : plafond"
    plafond = max((a.cout_maximal_usd for a in restants), default=0.0)
    return {
        "juge": juge,
        "methode": methode,
        "appels_prevus": len(prevus),
        "appels_deja_enregistres": len(prevus) - len(restants),
        "appels_restants": len(restants),
        "cout_moyen_par_appel_usd": moyen,
        "cout_prevu_restant_usd": moyen * len(restants),
        "cout_prevu_total_usd": moyen * len(prevus),
        "plafond_reserve_par_appel_usd": plafond,
        "budget_minimal_conseille_usd": moyen * len(restants) + plafond,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    analyse = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyse.add_argument("--juges", default=",".join(config.JUGES_COMPARES))
    mode = analyse.add_mutually_exclusive_group()
    mode.add_argument(
        "--recalculer", action="store_true", help="rejoue evals/cassettes/ sans appel réseau ; rapport partiel"
    )
    mode.add_argument("--estimer", action="store_true", help="coût prévu du run, sans aucun appel")
    options = analyse.parse_args(arguments)
    juges = [j.strip() for j in options.juges.split(",") if j.strip()]

    date = datetime.now(UTC).strftime("%Y-%m-%d")
    cas = charger_gold()
    hors_ligne = options.recalculer or options.estimer
    corpus = corpus_gold(cas, enregistrer=not hors_ligne)

    if options.estimer:
        mesures = couts_mesures()
        for juge in juges:
            estimation = estimer(juge, appels_prevus(juge, cas, corpus), mesures)
            print(json.dumps(estimation, ensure_ascii=False, indent=2))
        return 0
    refus: dict[str, object] = {}
    par_juge: dict[str, object] = {}
    # Un recalcul écrit dans son propre dossier : il n'écrase jamais le couts.json d'un run réel.
    prefixe = "recalcul" if options.recalculer else "evals"
    for juge in juges:
        dossier = RUNS_EVALS / f"{prefixe}-{date}" / re.sub(r"[^A-Za-z0-9_-]", "_", juge)
        resultats, non_evalues, ledger, interruption = evaluer_juge(juge, cas, corpus, dossier, options.recalculer)
        refus[juge] = {"par_role": refus_par_role(ledger), "par_sujet": refus_par_sujet(resultats)}
        par_juge[juge] = {
            "interruption_budget": interruption,
            "non_evalues": [n.model_dump(mode="json") for n in non_evalues],
            **metriques(resultats, ledger),
            "resultats": [
                {**r.model_dump(mode="json"), "correct": r.correct, "refus_fournisseur": r.refus_fournisseur}
                for r in resultats
            ],
        }

    rapport = {
        "date": date,
        "refus_fournisseur": refus,  # métrique de premier rang
        "lexiques_version": config.LEXIQUES_VERSION,
        "textclean_version": config.TEXTCLEAN_VERSION,
        "nb_cas": len(cas),
        "recalcul_depuis_cassettes": options.recalculer,
        "nb_cas_sujet_sensible": sum(c.sujet_sensible for c in cas),
        "nb_cas_ambigus": sum(c.ambigu for c in cas),
        "cas_valides_par_humain": sum(c.etiquetage.valide_par_humain for c in cas),
        "par_juge": par_juge,
    }
    suffixe = "_recalcul" if options.recalculer else ""
    sortie = Path(__file__).parent / f"rapport_{date}{suffixe}.json"
    sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nRapport : {sortie}\n\nRefus du fournisseur (cas sensibles / autres) :")
    for juge, detail in refus.items():
        assert isinstance(detail, dict)
        sujets = detail["par_sujet"]
        print(f"  {juge:<32} {sujets['sujets_sensibles']['taux']} / {sujets['autres_sujets']['taux']}")
    for juge, detail in par_juge.items():
        assert isinstance(detail, dict)
        if detail["alerte_faux_established"]:
            print(f"ALERTE [{juge}] : taux de faux ESTABLISHED {detail['taux_faux_established']:.1%}")
        if detail["alerte_aucune_confirmed"]:
            print(f"ALERTE [{juge}] : aucune vraie réfutation CONFIRMED")
        if detail["interruption_budget"]:
            print(f"INTERROMPU [{juge}] : {detail['interruption_budget']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
