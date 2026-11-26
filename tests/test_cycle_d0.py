"""Étape 3 : collecteurs sur réponses figées, gate du routeur, persistance, rapport, run complet."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from http_fige import RequeteNonFigeeError, client_fige  # noqa: E402

from metis import config  # noqa: E402
from metis.literature.arxiv import CollecteurArxiv  # noqa: E402
from metis.literature.pubmed import CollecteurPubMed  # noqa: E402
from metis.llm import ClientLLM, RunLedger  # noqa: E402
from metis.pipeline import cycle_d0, routeur  # noqa: E402
from metis.rapport.markdown import generer_rapport  # noqa: E402
from metis.schemas import (  # noqa: E402
    BaseLitterature,
    CandidatEquation,
    Claim,
    Drapeaux,
    EspeceCible,
    Grandeur,
    Incident,
    MotifRefus,
    Occurrence,
    Offsets,
    Porte,
    QuestionD0,
    RequetesInitiales,
    StatutClaim,
    Verdict,
    VerdictRoutage,
)

HTTP = Path(__file__).parent / "fixtures" / "http"
ENREGISTRER = os.environ.get("METIS_CASSETTE") in ("record", "record_missing")


# --------------------------------------------------------------------------- #
# Collecteurs (réponses réelles figées)
# --------------------------------------------------------------------------- #


def test_pubmed_pmcid_de_l_article_et_non_des_references() -> None:
    collecteur = CollecteurPubMed(client_fige(HTTP / "vitamine_d"), pause_s=0)
    souris, pmcid = collecteur.notice("22892342")
    assert pmcid == "PMC3477212"
    assert souris.texte_nettoye.startswith("[TITRE] Vitamin D supplementation protects against bone loss")
    _, sans_pmc = collecteur.notice("1331788")
    assert sans_pmc is None


def test_collecteur_sans_reseau_ni_fixture_echoue() -> None:
    collecteur = CollecteurPubMed(client_fige(HTTP / "vitamine_d"), pause_s=0)
    with pytest.raises(RequeteNonFigeeError):
        collecteur.notice("1")


def test_arxiv_resume_et_source_latex() -> None:
    collecteur = CollecteurArxiv(client_fige(HTTP / "arxiv_gompertz"), pause_s=0)
    resume, latex = collecteur.collecter("ti:gompertz AND ti:growth", 1)
    assert resume.source_id == "arxiv:1109.1269v2"
    assert resume.texte_nettoye.startswith("[TITRE] Gompertz and Verhulst frameworks for growth and decay")
    assert latex.source_id == "arxiv:1109.1269v2:tex"
    assert "\\begin{equation}" in latex.texte_nettoye  # maths conservées telles quelles
    assert "%% FICHIER" not in latex.texte_nettoye  # commentaires LaTeX retirés


# --------------------------------------------------------------------------- #
# Gate mécanique du routeur (0 LLM)
# --------------------------------------------------------------------------- #

EQUATION = "\\begin{equation} \\label {Malthus1diff}\n \\frac{dx}{dt}= r \\; x,   \n\n\\end{equation}"


def _corpus_arxiv() -> dict[str, object]:
    collecteur = CollecteurArxiv(client_fige(HTTP / "arxiv_gompertz"), pause_s=0)
    return {d.source_id: d for d in collecteur.collecter("ti:gompertz AND ti:growth", 1)}


def _etabli(claim_id: str = "c001") -> Claim:
    occurrence = (Occurrence(offsets=Offsets(debut=0, fin=50), context_window="fenêtre"),)
    claim = Claim.extrait(claim_id, "arxiv:1109.1269v2", "x" * 50).avec_ancrage(occurrence, 1)
    claim = claim.avec_drapeaux(Drapeaux(is_future_or_protocol=False))
    return claim.transitionner(StatutClaim.ESTABLISHED, Porte.AUDITEUR, "test")


def _candidat(**champs: str) -> CandidatEquation:
    base = {
        "source_id": "arxiv:1109.1269v2:tex",
        "citation_equation": EQUATION,
        "observable_nom": "taille de population normalisée x",
        "observable_unite": "sans dimension",
    }
    return CandidatEquation.model_validate({**base, **champs})


def test_gate_retient_une_equation_verbatim() -> None:
    corpus = _corpus_arxiv()
    resultat = routeur.gate([_etabli()], [], [_candidat()], corpus)  # type: ignore[arg-type]
    assert isinstance(resultat, tuple)
    candidat, ancrage = resultat
    assert candidat.citation_equation == EQUATION
    assert "Malthus model" in ancrage.context_window


@pytest.mark.parametrize(
    ("claims", "incidents", "candidats", "motif"),
    [
        ([], [], [_candidat()], MotifRefus.CLAIMS_INSUFFISANTS),
        ([_etabli()], [Incident(etape="contradicteur", cible="c001", motif="x")], [_candidat()],
         MotifRefus.CLAIMS_INSUFFISANTS),
        ([_etabli()], [], [], MotifRefus.PAS_D_EQUATION),
        ([_etabli()], [], [_candidat(citation_equation=EQUATION.replace("r \\;", "k \\;"))],
         MotifRefus.EQUATION_NON_VERBATIM),
        ([_etabli()], [], [_candidat(source_id="arxiv:0000.00000")], MotifRefus.EQUATION_NON_VERBATIM),
        ([_etabli()], [], [_candidat(citation_equation="which has the (catastrophic) exponential solution")],
         MotifRefus.PAS_D_EQUATION),
        ([_etabli()], [], [_candidat(observable_unite=" ")], MotifRefus.PAS_D_OBSERVABLE),
        ([_etabli()], [], [_candidat(citation_equation="inventée = pas dans le texte"),
                           _candidat(observable_nom="")], MotifRefus.PAS_D_OBSERVABLE),
    ],
)  # fmt: skip
def test_gate_refus_mecaniques(
    claims: list[Claim], incidents: list[Incident], candidats: list[CandidatEquation], motif: MotifRefus
) -> None:
    resultat = routeur.gate(claims, incidents, candidats, _corpus_arxiv())  # type: ignore[arg-type]
    assert resultat == VerdictRoutage(verdict=Verdict.REFUS, motif_refus=motif)


# --------------------------------------------------------------------------- #
# Persistance et rapport (0 LLM)
# --------------------------------------------------------------------------- #

QUESTION = QuestionD0(
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
    mesurande=Grandeur(nom="incidence des fractures (hanche, non vertébrales, toutes fractures)", unite="rapport de risques"),
    exclusions=("ostéoporose secondaire", "enfants et adolescents"),
    requetes_initiales=RequetesInitiales(pubmed=("vitamin D3 calcium hip fractures elderly women randomized",)),
)

FICHIERS_RUN = (
    "question.json", "requetes.json", "claims.json", "preuves_adverses.json", "verdict.json",
    "candidats_equation.json", "incidents.json", "rapport.md",
)  # fmt: skip


def _deps(dossier: Path, http: str) -> cycle_d0.Dependances:
    pause = config.PAUSE_NCBI_S if ENREGISTRER else 0
    return cycle_d0.Dependances(
        client=ClientLLM(config.mode_cassette(), config.dossier_cassettes()),
        ledger=RunLedger(dossier.name, dossier),
        collecteurs={
            BaseLitterature.PUBMED: CollecteurPubMed(client_fige(HTTP / http, ENREGISTRER), pause_s=pause),
            BaseLitterature.ARXIV: CollecteurArxiv(
                client_fige(HTTP / http, ENREGISTRER), pause_s=config.PAUSE_ARXIV_S if ENREGISTRER else 0
            ),
        },
        dossier_run=dossier,
    )


def test_persistance_et_rapport_reproductible_sans_llm(tmp_path: Path) -> None:
    dossier = tmp_path / "run-sans-llm"
    deps = _deps(dossier, "vitamine_d")
    corpus = tuple(CollecteurPubMed(client_fige(HTTP / "vitamine_d"), pause_s=0).notice(p)[0] for p in ("22892342",))
    etat = cycle_d0.demarrer(QUESTION, "run-sans-llm", deps, corpus)
    souris = "In a second study, female mice were pair-fed 30% EtOH diets with or without dietary supplementation"
    etat = cycle_d0._maj(etat, deps, claims=(Claim.extrait("c001", "pmid:22892342", souris),))
    etat = cycle_d0.filtrer(etat, deps)
    etat = cycle_d0.router(etat, deps)
    premier = generer_rapport(dossier)
    (dossier / "rapport.md").write_text(premier, encoding="utf-8")

    for nom in FICHIERS_RUN:
        assert (dossier / nom).exists(), nom
    assert (dossier / "corpus" / "pmid_22892342.json").exists()
    assert generer_rapport(dossier).encode() == premier.encode()
    assert "**refus** — motif `claims_insuffisants`" in premier
    assert "| c001 | [pmid:22892342](https://pubmed.ncbi.nlm.nih.gov/22892342/)" in premier
    assert "REJECTED_PREFILTER | prefilter | espece_exclue_sans_marqueur_humain: mice |" in premier
    for titre in ("## 1. Question", "## 2. Verdict exécutif", "## 3. Claims ESTABLISHED", "## 4. Claims CONTRADICTED",
                  "## 5. Cimetière", "## 6. Requêtes adverses", "## 7. Bilan financier"):  # fmt: skip
        assert titre in premier


# --------------------------------------------------------------------------- #
# Run complet sur cassettes
# --------------------------------------------------------------------------- #


def test_cycle_complet_sur_cassettes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_RESULTATS_REQUETE_INITIALE", 3)
    monkeypatch.setattr(config, "MAX_RESULTATS_REQUETE_ADVERSE", 2)
    dossier = tmp_path / "cycle-d0"
    deps = _deps(dossier, "cycle_vitamine_d")

    etat = cycle_d0.executer_d0(QUESTION, "cycle-d0", deps)

    for nom in (*FICHIERS_RUN, "couts.json"):
        assert (dossier / nom).exists(), nom
    assert len(list((dossier / "corpus").iterdir())) == len(etat.documents) > 0
    assert len(list((dossier / "reponses_llm").iterdir())) == len(deps.ledger.entrees) > 0
    assert etat.verdict is not None
    rapport = (dossier / "rapport.md").read_bytes()
    assert generer_rapport(dossier).encode() == rapport  # reproductible à l'octet
    assert all(c.statut is not StatutClaim.EXTRACTED for c in etat.claims)  # aucun claim laissé en route
    # Le run doit avoir réellement jugé : des claims extraits et audités, aucun échec de juge.
    assert etat.claims, "aucun claim extrait"
    assert any(e.role.value == "auditeur" for e in deps.ledger.entrees), "aucun appel à l'auditeur"
    echecs_juge = ("juge_sortie_invalide", "refus_fournisseur")
    motifs = [i.motif.split(" ")[0] for i in etat.incidents]
    motifs += [t.motif or "" for c in etat.claims for t in c.historique]
    motifs += [t.motif or "" for p in etat.preuves for t in p.historique]
    assert not [m for m in motifs if m in echecs_juge], etat.incidents
    if not ENREGISTRER:
        assert all(e.cassette for e in deps.ledger.entrees)


# --------------------------------------------------------------------------- #
# Gold set (0 LLM : cohérence et part déterministe du pré-filtre)
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

import run_evals  # noqa: E402

from metis.literature.prefilter import prefiltrer_claim  # noqa: E402
from metis.literature.verify import verifier_claim  # noqa: E402

CATEGORIES_SPEC = {
    "transposition_animale", "in_vitro", "protocole_sans_resultat", "negation_fenetre",
    "mesurande_voisin", "vraie_refutation", "fausse_refutation",
}  # fmt: skip


def test_gold_set_reel_et_verbatim() -> None:
    cas = run_evals.charger_gold()
    assert len(cas) >= 40
    assert CATEGORIES_SPEC <= {c.categorie for c in cas}
    corpus = run_evals.corpus_gold(cas)  # notices réelles figées, aucune requête réseau
    for c in cas:
        citations = [(c.source_id, c.citation)] if isinstance(c, run_evals.CasClaim) else [
            (c.claim.source_id, c.claim.citation), (c.preuve.source_id, c.preuve.citation)]  # fmt: skip
        for source_id, citation in citations:
            claim, _ = verifier_claim(Claim.extrait("c001", source_id, citation), corpus)
            assert claim.statut is StatutClaim.EXTRACTED, (c.cas_id, claim.motif_rejet)


def test_gold_set_sujets_sensibles() -> None:
    cas = run_evals.charger_gold()
    assert sum(c.sujet_sensible for c in cas) >= 25
    mtor = [c for c in cas if c.question.question_id == "gold-mtor-vaccin"]
    assert len(mtor) == 15 and all(c.sujet_sensible for c in mtor)


def test_compteurs_refus_fournisseur(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from metis.schemas import EntreeLedger, RoleLLM

    ledger = RunLedger("evals", tmp_path)
    for role, refus in ((RoleLLM.AUDITEUR, True), (RoleLLM.AUDITEUR, False), (RoleLLM.EXTRACTEUR, True)):
        ledger.enregistrer(EntreeLedger(
            role=role, model="m", prompt_tokens=1, completion_tokens=0, cost_usd=0.0, latency_ms=0.0,
            cassette=False, horodatage=datetime.now(UTC), refus_fournisseur=refus))  # fmt: skip
    assert run_evals.refus_par_role(ledger) == {
        "auditeur": {"refus": 1, "total": 2, "taux": 0.5},
        "extracteur": {"refus": 1, "total": 1, "taux": 1.0},
    }
    resultats = [
        run_evals.ResultatCas(cas_id=f"g{i}", type="claim", categorie="x", sujet_sensible=sensible, ambigu=False,
                              attendus=("ESTABLISHED",), obtenu="REJECTED_SEMANTIC", motif=motif)
        for i, (sensible, motif) in enumerate([(True, "refus_fournisseur"), (True, "m"), (False, "m")])
    ]  # fmt: skip
    assert run_evals.refus_par_sujet(resultats) == {
        "sujets_sensibles": {"refus": 1, "total": 2, "taux": 0.5},
        "autres_sujets": {"refus": 0, "total": 1, "taux": 0.0},
    }


def test_gold_set_ambigus_exclus_des_metriques(tmp_path: Path) -> None:
    cas = run_evals.charger_gold()
    ambigus = {c.cas_id for c in cas if c.ambigu}
    assert ambigus and all(c.categorie == "vraie_refutation" for c in cas if c.ambigu)
    assert all(c.question.intervention.strip() and c.question.population.strip() for c in cas)
    resultats = [
        run_evals.ResultatCas(cas_id="a", type="claim", categorie="x", sujet_sensible=False, ambigu=True,
                              attendus=("REJECTED_SEMANTIC",), obtenu="ESTABLISHED", motif="m"),
        run_evals.ResultatCas(cas_id="b", type="claim", categorie="x", sujet_sensible=False, ambigu=False,
                              attendus=("REJECTED_SEMANTIC",), obtenu="REJECTED_SEMANTIC", motif="m"),
    ]  # fmt: skip
    m = run_evals.metriques(resultats, RunLedger("evals", tmp_path))
    assert m["taux_faux_established"] == 0.0  # le faux ESTABLISHED ambigu ne compte pas
    liste_ambigus = m["cas_ambigus"]
    assert isinstance(liste_ambigus, list)
    assert m["nb_cas_mesures"] == 1 and [a["cas_id"] for a in liste_ambigus] == ["a"]


def test_alerte_aucune_confirmed(tmp_path: Path) -> None:
    def cas(obtenu: str) -> run_evals.ResultatCas:
        return run_evals.ResultatCas(cas_id="r", type="refutation", categorie="vraie_refutation",
                                     sujet_sensible=False, ambigu=False, attendus=("CONFIRMED",),
                                     obtenu=obtenu, motif="m")  # fmt: skip

    ledger = RunLedger("evals", tmp_path)
    assert run_evals.metriques([cas("REJECTED_SEMANTIC")], ledger)["alerte_aucune_confirmed"] is True
    assert run_evals.metriques([cas("CONFIRMED")], ledger)["alerte_aucune_confirmed"] is False


def test_gold_set_vraies_refutations_propres() -> None:
    propres = [c for c in run_evals.charger_gold() if c.categorie == "vraie_refutation" and not c.ambigu]
    assert len(propres) >= 1
    assert all(isinstance(c, run_evals.CasRefutation) and c.statuts_acceptables == (StatutPreuve.CONFIRMED,)
               for c in propres)  # fmt: skip


def test_budget_evals_un_dollar_par_juge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.ENV_BUDGET, raising=False)
    assert run_evals.budget_par_juge() == 1.0
    monkeypatch.setenv(config.ENV_BUDGET, "3")
    assert run_evals.budget_par_juge() == 3.0


def test_gold_set_part_deterministe_du_prefiltre() -> None:
    """Le pré-filtre rejette exactement les cas étiquetés REJECTED_PREFILTER, aucun autre."""
    cas = [c for c in run_evals.charger_gold() if isinstance(c, run_evals.CasClaim)]
    corpus = run_evals.corpus_gold(cas)  # type: ignore[arg-type]
    for c in cas:
        claim, _ = verifier_claim(Claim.extrait("c001", c.source_id, c.citation), corpus)
        filtre = prefiltrer_claim(claim, c.question)
        attendu_prefiltre = c.statuts_acceptables == (StatutClaim.REJECTED_PREFILTER,)
        assert (filtre.statut is StatutClaim.REJECTED_PREFILTER) == attendu_prefiltre, (c.cas_id, filtre.motif_rejet)


# --------------------------------------------------------------------------- #
# Gardes mécaniques de l'auditeur (0 LLM)
# --------------------------------------------------------------------------- #

from metis.pipeline.auditeur import contient_resultat_chiffre, statut_claim, statut_preuve  # noqa: E402
from metis.schemas import AxeJugement, EvaluationFenetre, EvaluationFenetrePreuve, StatutPreuve  # noqa: E402


CITATION_MYOSITE_DOSE = (
    "This phase IIb/III double-blind, randomised, controlled trial (RCT) of sirolimus involves 140 IBM patients "
    "randomly assigned with equal allocation to sirolimus (2 mg) or matching placebo."
)


@pytest.mark.parametrize(
    ("citation", "chiffre"),
    [
        ("decreased from 13.2 +/- 2.2 percent to 10.5 +/- 1.6 percent", True),
        ("mean change 0.50 ± 4.80", True),
        ("significantly different (p<0.05)", True),
        ("P = .03 for the comparison", True),
        ("HR = 1.52 versus placebo", True),
        ("odds ratio [OR] 0·601 [90% CI 0·391-0·922]", True),
        ("RR, 1.15 for falls", True),
        ("relative risk 0.78 (95% confidence interval 0.61 to 0.99)", True),
        ("MD -0.4 points versus placebo", True),
        ("IC à 95 % : 0,61 à 0,99", True),
        ("RAD001 enhanced the response to the influenza vaccine by about 20%", False),
        ("the patients lost 2.7 kg of weight", False),
        ("25(OH)D rose to 99.7 nmol/L", False),
        ("The trial has 80% power to detect a 20% reduction", False),
        ("a power of 90% at p < 0.05 to detect a difference", False),
        ("Secondary outcomes will evaluate efficacy and safety to inform future clinical trial design.", False),
        ("ANZCTR: ACTRN12620001226998p, ClinicalTrials.gov: NCT04789070, UTN: U1111-1258-1354", False),
        ("the trial will assess whether risk OR benefit predominates", False),
        (CITATION_MYOSITE_DOSE, False),
    ],
)
def test_resultat_chiffre(citation: str, chiffre: bool) -> None:
    assert contient_resultat_chiffre(citation) is chiffre


AXES_CLAIM = ("espece_et_modele", "temporalite_et_statut_de_preuve", "mesurande", "population", "intervention")


def _axe(verdict: str = "conforme") -> AxeJugement:
    return AxeJugement.model_validate({"verdict": verdict, "justification": "test"})


def _evaluation(statut: str = "ESTABLISHED", **axes: str) -> EvaluationFenetre:
    return EvaluationFenetre.model_validate(
        {"indice_contexte": 0, **{n: _axe(axes.get(n, "conforme")) for n in AXES_CLAIM}, "statut": statut,
         "motif": "m"}
    )  # fmt: skip


def _claim_drapeau(citation: str, futur: bool) -> Claim:
    occurrence = (Occurrence(offsets=Offsets(debut=0, fin=len(citation)), context_window=citation),)
    claim = Claim.extrait("c001", "pmid:1", citation).avec_ancrage(occurrence, 1)
    return claim.avec_drapeaux(Drapeaux(is_future_or_protocol=futur))


@pytest.mark.parametrize(
    ("citation", "futur", "attendu"),
    [
        ("Secondary outcomes will evaluate efficacy and safety to inform future trial design.", True,
         StatutClaim.CLAIMED_UNPROVEN),
        (CITATION_MYOSITE_DOSE, True, StatutClaim.CLAIMED_UNPROVEN),  # « 2 mg » n'est pas un résultat
        ("This trial, registered as NCT04789070, improved the vaccine response by about 20%.", True,
         StatutClaim.CLAIMED_UNPROVEN),
        ("This trial, registered as NCT04789070, lowered fracture risk (HR 0.76; 95% CI 0.61 to 0.95).", True,
         StatutClaim.ESTABLISHED),
        ("we observed an improvement in the response to influenza vaccination in this group", False,
         StatutClaim.ESTABLISHED),
    ],
)  # fmt: skip
def test_garde_futur_sans_resultat_chiffre(citation: str, futur: bool, attendu: StatutClaim) -> None:
    statut, motif = statut_claim(_evaluation(), _claim_drapeau(citation, futur))
    assert statut is attendu
    if attendu is StatutClaim.CLAIMED_UNPROVEN:
        assert "sans résultat chiffré" in motif


@pytest.mark.parametrize(
    ("statut_juge", "axes", "attendu"),
    [
        # population ou intervention non conforme ou non vérifiable : jamais CLAIMED_UNPROVEN
        ("ESTABLISHED", {"population": "non_conforme"}, StatutClaim.REJECTED_SEMANTIC),
        ("ESTABLISHED", {"population": "indetermine"}, StatutClaim.REJECTED_SEMANTIC),
        ("ESTABLISHED", {"intervention": "non_conforme"}, StatutClaim.REJECTED_SEMANTIC),
        ("CLAIMED_UNPROVEN", {"population": "indetermine"}, StatutClaim.REJECTED_SEMANTIC),
        ("CLAIMED_UNPROVEN", {"intervention": "indetermine", "temporalite_et_statut_de_preuve": "non_conforme"},
         StatutClaim.REJECTED_SEMANTIC),
        ("ESTABLISHED", {"mesurande": "non_conforme"}, StatutClaim.REJECTED_SEMANTIC),
        # CLAIMED_UNPROVEN réservé à l'axe temporalité
        ("ESTABLISHED", {"temporalite_et_statut_de_preuve": "indetermine"}, StatutClaim.CLAIMED_UNPROVEN),
        ("CLAIMED_UNPROVEN", {"temporalite_et_statut_de_preuve": "non_conforme"}, StatutClaim.CLAIMED_UNPROVEN),
        ("CLAIMED_UNPROVEN", {}, StatutClaim.REJECTED_SEMANTIC),
        # le juge peut toujours rejeter
        ("REJECTED_SEMANTIC", {}, StatutClaim.REJECTED_SEMANTIC),
        ("ESTABLISHED", {}, StatutClaim.ESTABLISHED),
    ],
)  # fmt: skip
def test_regles_de_statut_claim(statut_juge: str, axes: dict[str, str], attendu: StatutClaim) -> None:
    claim = _claim_drapeau("Hip fractures were 43 percent lower (P = 0.043) with vitamin D3 and calcium.", False)
    assert statut_claim(_evaluation(statut_juge, **axes), claim)[0] is attendu


@pytest.mark.parametrize("axe", ["population", "intervention", "force_de_refutation", "espece_et_modele"])
def test_garde_axes_preuve(axe: str) -> None:
    noms = (*AXES_CLAIM, "force_de_refutation")
    evaluation = EvaluationFenetrePreuve.model_validate(
        {"indice_contexte": 0, **{n: _axe("non_conforme" if n == axe else "conforme") for n in noms},
         "statut": "CONFIRMED", "motif": "m"}
    )  # fmt: skip
    assert statut_preuve(evaluation)[0] is StatutPreuve.REJECTED_SEMANTIC


# --------------------------------------------------------------------------- #
# Contexte transmis à l'auditeur (0 LLM)
# --------------------------------------------------------------------------- #

from metis.literature.textclean import construire_document  # noqa: E402
from metis.literature.verify import verifier_claim as _verifier  # noqa: E402
from metis.pipeline.auditeur import baliser, contextes  # noqa: E402
from metis.schemas import Document, FormatSource  # noqa: E402


def test_balisage_des_occurrences() -> None:
    assert baliser("abc abc xyz", [(4, 7), (0, 3)]) == "<<<abc>>> <<<abc>>> xyz"
    assert baliser("abcdef", [(0, 4), (2, 6)]) == "<<<abcd>>>ef"  # chevauchement ignoré


def test_abstract_court_envoye_en_entier_et_balise() -> None:
    corpus = {d.source_id: d for d in (CollecteurPubMed(client_fige(HTTP / "vitamine_d"), pause_s=0).notice("1331788")[0],)}
    document = corpus["pmid:1331788"]
    assert len(document.texte_nettoye) < config.SEUIL_DOCUMENT_ENTIER
    citation = "Supplementation with vitamin D3 and calcium reduces the risk of hip fractures"
    claim, _ = _verifier(Claim.extrait("c001", "pmid:1331788", citation), corpus)
    (seul,), resume = contextes(claim, document)
    assert resume is None and seul["type"] == "document_entier"
    texte = str(seul["texte"])
    assert f"<<<{citation}>>>" in texte
    assert "3270 healthy ambulatory women" in texte  # la population, loin de la citation, est visible
    assert texte.replace("<<<", "").replace(">>>", "") == document.texte_nettoye


def test_texte_integral_fenetres_et_resume() -> None:
    xml = (FIXTURES_TEXTCLEAN / "pmc_jats_minimal.xml").read_text(encoding="utf-8")
    long = xml.replace("<p>Values are means.</p>", "<p>" + "Values are means. " * 400 + "</p>")
    document = construire_document("pmc:PMC1", FormatSource.PMC_XML, long)
    assert len(document.texte_nettoye) >= config.SEUIL_DOCUMENT_ENTIER
    claim, _ = _verifier(Claim.extrait("c001", "pmc:PMC1", "Plasma levels reached 10−3 M and the H2O content"), {document.source_id: document})
    liste, resume = contextes(claim, document)
    assert [c["type"] for c in liste] == ["fenetre"]
    assert resume == "[SECTION] Background\n\nRapamycin extends lifespan in mice."


def test_contexte_exige_le_bon_document() -> None:
    autre = Document(source_id="pmid:2", format=FormatSource.PUBMED_XML, texte_brut="", texte_nettoye="x",
                     textclean_version="2")  # fmt: skip
    with pytest.raises(ValueError):
        contextes(_etabli(), autre)


FIXTURES_TEXTCLEAN = Path(__file__).parent / "fixtures" / "textclean"


# --------------------------------------------------------------------------- #
# Invariant : seul l'auditeur fait passer un claim en ESTABLISHED (0 LLM)
# --------------------------------------------------------------------------- #

import ast  # noqa: E402
import importlib  # noqa: E402
import pkgutil  # noqa: E402

import metis  # noqa: E402

RACINE_METIS = Path(metis.__file__).parent


def _appels_transitionner_vers_established(arbre: ast.AST) -> list[int]:
    lignes = []
    for noeud in ast.walk(arbre):
        if (
            isinstance(noeud, ast.Call)
            and isinstance(noeud.func, ast.Attribute)
            and noeud.func.attr == "transitionner"
            and noeud.args
            and isinstance(noeud.args[0], ast.Attribute)
            and noeud.args[0].attr == "ESTABLISHED"
        ):
            lignes.append(noeud.lineno)
    return lignes


def test_metis_n_expose_aucune_injection() -> None:
    for module in pkgutil.walk_packages([str(RACINE_METIS)], prefix="metis."):
        noms = {n.lower() for n in dir(importlib.import_module(module.name))}
        assert not {n for n in noms if "inject" in n}, module.name
    for fichier in RACINE_METIS.rglob("*.py"):
        texte = fichier.read_text(encoding="utf-8")
        assert "injecté pour tester" not in texte, fichier


def test_seul_l_auditeur_etablit_un_claim() -> None:
    """Aucun module de metis ne transitionne en StatutClaim.ESTABLISHED littéralement ; l'auditeur
    le fait via le statut issu de statut_claim(), après ses règles."""
    for fichier in RACINE_METIS.rglob("*.py"):
        arbre = ast.parse(fichier.read_text(encoding="utf-8"))
        assert not _appels_transitionner_vers_established(arbre), fichier
    auditeur_src = (RACINE_METIS / "pipeline" / "auditeur.py").read_text(encoding="utf-8")
    assert "claim.transitionner(statut, Porte.AUDITEUR, motif)" in auditeur_src


def test_detecteur_d_injection_fonctionne() -> None:
    aide = Path(__file__).parent / "fixtures" / "injection.py"
    assert _appels_transitionner_vers_established(ast.parse(aide.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# run_evals : recalcul partiel et estimation, sans appel
# --------------------------------------------------------------------------- #


def test_recalcul_sans_cassette_liste_les_non_evalues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_evals, "CASSETTES_EVALS", tmp_path / "vide")
    cas = [c for c in run_evals.charger_gold() if c.cas_id in ("g001", "g003")]  # auditeur / pré-filtre
    corpus = run_evals.corpus_gold(cas)
    resultats, non_evalues, ledger, interruption = run_evals.evaluer_juge(
        config.MODELE_JUGE_DEFAUT, cas, corpus, tmp_path / "run", recalculer=True
    )
    assert [r.cas_id for r in resultats] == ["g003"] and resultats[0].obtenu == "REJECTED_PREFILTER"
    assert [n.cas_id for n in non_evalues] == ["g001"] and interruption is None
    assert ledger.entrees == ()


def test_estimation_depuis_les_couts_mesures(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from metis.schemas import EntreeLedger, RoleLLM

    ledger = RunLedger("evals", tmp_path / "evals-x" / "juge")
    for cout, cassette in ((0.02, False), (0.04, False), (9.0, True)):  # le rejeu n'est pas compté
        ledger.enregistrer(EntreeLedger(
            role=RoleLLM.AUDITEUR, model="m/a", prompt_tokens=1000, completion_tokens=100, cost_usd=cout,
            latency_ms=0.0, cassette=cassette, horodatage=datetime.now(UTC)))  # fmt: skip
    mesures = run_evals.couts_mesures(tmp_path)
    assert mesures[("m/a", "auditeur")]["appels"] == 2
    assert mesures[("m/a", "auditeur")]["cout_usd"] == pytest.approx(0.03)
    prevus = [
        run_evals.AppelPrevu(cas_id=f"g{i}", role=RoleLLM.AUDITEUR, deja_enregistre=i < 2, cout_maximal_usd=0.3)
        for i in range(5)
    ]
    estimation = run_evals.estimer("m/a", prevus, mesures)
    assert estimation["appels_restants"] == 3
    assert estimation["cout_prevu_restant_usd"] == pytest.approx(0.09)
    autre = run_evals.estimer(config.MODELE_JUGE_DEFAUT, prevus, mesures)  # tokens mesurés × tarif
    t = config.TABLE_PRIX[config.MODELE_JUGE_DEFAUT]
    assert autre["cout_moyen_par_appel_usd"] == pytest.approx((1000 * t.entree_usd_par_mtok + 100 * t.sortie_usd_par_mtok) / 1e6)


def test_appels_prevus_sans_reseau(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_evals, "CASSETTES_EVALS", tmp_path)
    cas = [c for c in run_evals.charger_gold() if c.cas_id in ("g001", "g003", "g062")]
    prevus = run_evals.appels_prevus(config.MODELE_JUGE_DEFAUT, cas, run_evals.corpus_gold(cas))
    assert [a.cas_id for a in prevus] == ["g001", "g062"]  # g003 tranché par le pré-filtre
    assert not any(a.deja_enregistre for a in prevus)
