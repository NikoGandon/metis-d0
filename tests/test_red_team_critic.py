"""Red team du critique D0 : 10 notices PubMed réelles figées (vitamine D et fractures).

Questions humaines in vivo ; chaque piège est une citation injectée après l'extracteur
(texte réel du document, sauf la citation inventée) pour éprouver les portes de façon
déterministe : verify.py, prefilter.py, puis l'auditeur (cassettes) et le routeur.
La question myosite (PMID 40018746) est conservée telle quelle.

Domaine vitamine D choisi après l'enregistrement du 2026-11-03 : sur le corpus
Rapamycine / vaccin antigrippal, le fournisseur refusait 14 réponses sur 16
(native_finish_reason=refusal). Ces refus restent fail-closed (refus_fournisseur),
sans retry ni repli ; ils sont mesurés par evals/run_evals.py.

Corpus figé le 2026-11-04 par :
    .venv/bin/python tests/fixtures/figer_corpus.py notices vitamine_d 1331788 9278463 \\
        22892342 19442605 42346679 32193264 40492704 18433305 32492153 25623291
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from http_fige import client_fige  # noqa: E402
from injection import MOTIF_INJECTION, injecter_etabli  # noqa: E402

from metis import config  # noqa: E402
from metis.literature.pubmed import CollecteurPubMed  # noqa: E402
from metis.llm import ClientLLM, RunLedger  # noqa: E402
from metis.pipeline import cycle_d0  # noqa: E402
from metis.pipeline.auditeur import axes_non_conformes  # noqa: E402
from metis.schemas import (  # noqa: E402
    Claim,
    Document,
    EspeceCible,
    EtatD0,
    Grandeur,
    MotifRefus,
    Porte,
    PreuveAdverse,
    QuestionD0,
    RequetesInitiales,
    StatutClaim,
    StatutPreuve,
    Verdict,
)

FIXTURES = Path(__file__).parent / "fixtures"
PMIDS = (
    "1331788", "9278463", "22892342", "19442605", "42346679",
    "32193264", "40492704", "18433305", "32492153", "25623291",
)  # fmt: skip
PMID_HORS_CORPUS = "12609940"  # réel (Trivedi et al. 2003, vitamine D3 et fractures), absent du corpus
PMID_MYOSITE = "40018746"  # figé dans tests/fixtures/http/gold/

QUESTION = QuestionD0(
    question_id="vitd-fractures-ages",
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
    requetes_initiales=RequetesInitiales(pubmed=("vitamin D calcium fractures elderly randomized",)),
)

QUESTION_MYOSITE = QuestionD0(
    question_id="sirolimus-myosite",
    sous_question=(
        "Chez les patients atteints de myosite à inclusions, le sirolimus ralentit-il la progression "
        "fonctionnelle de la maladie ?"
    ),
    espece_cible=EspeceCible.HUMAN_IN_VIVO,
    population="adultes de plus de 40 ans atteints de myosite à inclusions ; sirolimus 2 mg/j contre placebo",
    intervention="sirolimus oral 2 mg/j contre placebo",
    mesurande=Grandeur(nom="variation du score IBM Functional Rating Scale (IBM-FRS)", unite="points"),
    requetes_initiales=RequetesInitiales(pubmed=("sirolimus inclusion body myositis",)),
)

# Vraie réfutation propre : même population (femmes ménopausées américaines), même
# intervention (œstrogène + progestatif oral), même mesurande (maladie coronaire).
# Nurses' Health Study 1996 (observationnel, RR 0,39) contre WHI 2002 (essai, HR 1,29).
QUESTION_THS = QuestionD0(
    question_id="ths-coronaire",
    sous_question=(
        "Chez la femme ménopausée sans maladie coronaire connue, un traitement hormonal associant œstrogène et "
        "progestatif réduit-il le risque de maladie coronaire ?"
    ),
    espece_cible=EspeceCible.HUMAN_IN_VIVO,
    population="femmes ménopausées sans maladie coronaire connue, suivies aux États-Unis",
    intervention="œstrogène associé à un progestatif par voie orale, contre absence de traitement hormonal ou placebo",
    mesurande=Grandeur(nom="maladie coronaire (infarctus non fatal ou décès coronarien)", unite="risque relatif"),
    requetes_initiales=RequetesInitiales(pubmed=("estrogen progestin coronary heart disease postmenopausal",)),
)
PMIDS_THS = ("8672166", "12117397")

S = StatutClaim


@dataclass(frozen=True)
class Piege:
    """attendus : statut final. Le motif doit aussi être le bon :
    - porte : la porte qui a statué en dernier ;
    - motif_exact : pour les gates déterministes, le motif mot pour mot ;
    - axes : motif principal ; pour l'auditeur, au moins un de ces axes doit être non
      conforme, d'autres axes justifiés peuvent s'y ajouter (ensemble vide : tous les
      axes conformes, cas des contrôles positifs).
    """

    nom: str
    source_id: str
    citation: str
    attendus: frozenset[StatutClaim]
    porte: Porte
    motif_exact: str | None = None
    axes: frozenset[str] = frozenset()
    question: QuestionD0 = QUESTION


AUCUN_AXE: frozenset[str] = frozenset()
TEMPORALITE = frozenset({"temporalite_et_statut_de_preuve"})


PIEGES = (
    Piege("etabli_chapuy", "pmid:1331788",
          "the number of hip fractures was 43 percent lower (P = 0.043) and the total number of nonvertebral "
          "fractures was 32 percent lower (P = 0.015) among the women treated with vitamin D3 and calcium than "
          "among those who received placebo",
          frozenset({S.ESTABLISHED}), Porte.AUDITEUR, axes=AUCUN_AXE),
    Piege("etabli_dawson_hughes", "pmid:9278463",
          "Of 37 subjects who had nonvertebral fractures, 26 were in the placebo group and 11 were in the "
          "calcium-vitamin D group (P=0.02).",
          frozenset({S.ESTABLISHED}), Porte.AUDITEUR, axes=AUCUN_AXE),
    Piege("souris", "pmid:22892342",
          "In a second study, female mice were pair-fed 30% EtOH diets with or without dietary supplementation "
          "with vitamin D3 (cholecalciferol; VitD) for 40 days.",
          frozenset({S.REJECTED_PREFILTER}), Porte.PREFILTER, "espece_exclue_sans_marqueur_humain: mice"),
    Piege("rats", "pmid:19442605",
          "Female SD rats of six-month-old (n=40) allocated randomly into two groups were given ovariectomy.",
          frozenset({S.REJECTED_PREFILTER}), Porte.PREFILTER, "espece_exclue_sans_marqueur_humain: rats"),
    Piege("in_vitro", "pmid:42346679",
          "Using our in vitro model, we compared the osteogenic behaviour of primary healthy human osteoblasts "
          "(hOBs) and primary osteoporotic human osteoblasts (hopOBs)",
          frozenset({S.REJECTED_PREFILTER}), Porte.PREFILTER, "espece_exclue_sans_marqueur_humain: in vitro"),
    # Protocole EXVITD, enregistré, sans résultat. Intervention hors périmètre (vitamine D3 seule +
    # entraînement, sans calcium) : la règle donne REJECTED_SEMANTIC, mais le motif principal du piège
    # (protocole : axe temporalité non conforme) doit rester présent.
    Piege("nct_sans_resultat", "pmid:32193264",
          "The study is being conducted according to the principles of the Declaration of Helsinki.Trial "
          "registration numberNCT02467153",
          frozenset({S.REJECTED_SEMANTIC}), Porte.AUDITEUR, axes=TEMPORALITE),
    Piege("futur", "pmid:32193264",
          "muscle function (short physical performance battery, timed up and go), falls and fractures as events "
          "will be assessed",
          frozenset({S.REJECTED_SEMANTIC}), Porte.AUDITEUR, axes=TEMPORALITE),
    # Contexte : « None of the three treatments reduced the IR of total VFs overall, however, » —
    # la réduction citée est celle de l'exercice, pas de la vitamine D (axe intervention justifié).
    Piege("negation_fenetre", "pmid:40492704",
          "the IR was reduced with SHEP compared to the control exercise program in women (IR ratio 0.52, 95% CI "
          "0.28, 0.98)",
          frozenset({S.REJECTED_SEMANTIC}), Porte.AUDITEUR,
          axes=frozenset({"temporalite_et_statut_de_preuve", "mesurande", "intervention"})),
    # Même intervention et même mesurande, mais recrues de la Navy de 17 à 35 ans.
    Piege("population_differente", "pmid:18433305",
          "we found that the calcium and vitamin D group had a 20% lower incidence of SFx than the control group "
          "(5.3% versus 6.6%, respectively, p = 0.0026 for Fisher's exact test)",
          frozenset({S.REJECTED_SEMANTIC}), Porte.AUDITEUR, axes=frozenset({"population"})),
    Piege("citation_inventee", "pmid:1331788",
          "Vitamin D3 and calcium halved the number of hip fractures in every elderly woman studied.",
          frozenset({S.REJECTED_VERBATIM}), Porte.VERIFY, "citation_non_verbatim"),
    Piege("pmid_hors_corpus", f"pmid:{PMID_HORS_CORPUS}",
          "Relative risks in the vitamin D group compared with the placebo group were 0.78 (95% confidence interval "
          "0.61 to 0.99, P=0.04) for any first fracture",
          frozenset({S.REJECTED_VERBATIM}), Porte.VERIFY, "source_hors_corpus"),
    # Myosite : question et pièges inchangés.
    Piege("myosite_nct_sans_resultat", f"pmid:{PMID_MYOSITE}",
          "ANZCTR: ACTRN12620001226998p, ClinicalTrials.gov: NCT04789070, UTN: U1111-1258-1354",
          frozenset({S.CLAIMED_UNPROVEN}), Porte.AUDITEUR, axes=TEMPORALITE, question=QUESTION_MYOSITE),
    # Aucune phrase du protocole ne porte au futur sur l'IBM-FRS (la seule qui le cite est un
    # objectif : « This RCT aims to assess… »). Citation gardée : elle porte sur les critères
    # secondaires (mesurande non conforme) ; la temporalité doit rester parmi les axes.
    Piege("myosite_futur", f"pmid:{PMID_MYOSITE}",
          "Secondary outcomes will evaluate efficacy and safety to inform future clinical trial design.",
          frozenset({S.REJECTED_SEMANTIC}), Porte.AUDITEUR, axes=TEMPORALITE, question=QUESTION_MYOSITE),
    # Claim cible de la vraie réfutation, injecté en ESTABLISHED (voir MOTIF_INJECTION) puis
    # contredit par la preuve WHI p001. L'auditeur le rejette à juste titre (étude
    # d'observation, population et intervention non vérifiables) : seul le chemin de
    # réfutation est testé ici.
    Piege("etabli_nhs_refute", "pmid:8672166",
          "We observed a marked decrease in the risk of major coronary heart disease among women who took estrogen "
          "with progestin (multivariate adjusted relative risk, 0.39; 95 percent confidence interval, 0.19 to 0.78)",
          frozenset({S.CONTRADICTED}), Porte.CONTRADICTEUR, "p001", question=QUESTION_THS),
)  # fmt: skip

MOTIFS_ECHEC_JUGE = {"juge_sortie_invalide", "refus_fournisseur"}
# Motif principal de la fausse réfutation : force de réfutation ou mesurande (chutes, pas
# fractures). Population et intervention (VITAL : vitamine D3 seule) peuvent s'y ajouter.
AXES_FAUSSE_REFUTATION = frozenset({"force_de_refutation", "mesurande"})


def _ecarts_motif(piege: Piege, claim: Claim) -> str | None:
    """Ce qui ne va pas dans le motif final du claim, ou None."""
    derniere = claim.historique[-1]
    if derniere.porte is not piege.porte:
        return f"porte {derniere.porte.value} au lieu de {piege.porte.value}"
    if piege.motif_exact is not None:
        return None if derniere.motif == piege.motif_exact else f"motif {derniere.motif!r}"
    # porte auditeur : motif principal lu dans les axes
    axes = axes_non_conformes(derniere.motif)
    if piege.axes == AUCUN_AXE:
        return None if not axes else f"axes non conformes {sorted(axes)}"
    return None if axes & piege.axes else f"axes non conformes {sorted(axes)}, attendu l'un de {sorted(piege.axes)}"


# Partage les mots-clés du claim établi (vitamine D, placebo, adultes âgés) mais porte sur
# les chutes, pas sur les fractures.
FAUSSE_REFUTATION = PreuveAdverse.candidate(
    "p001",
    "c001",
    "vitamin D supplementation older adults no benefit",
    "pmid:32492153",
    "Daily supplemental vitamin D3 vs placebo did not decrease fall risk in generally healthy adults not selected "
    "for vitamin D insufficiency.",
)

VRAIE_REFUTATION = PreuveAdverse.candidate(
    "p001",
    "c001",
    "estrogen plus progestin coronary heart disease randomized trial",
    "pmid:12117397",
    "Estimated hazard ratios (HRs) (nominal 95% confidence intervals [CIs]) were as follows: CHD, 1.29 (1.02-1.63) "
    "with 286 cases",
)

# Pièges jugés après les gates (auditeur, puis contradicteur) : EXTRACTED à la sortie des gates.
SEMANTIQUES = {p.nom for p in PIEGES if p.porte in (Porte.AUDITEUR, Porte.CONTRADICTEUR)}


def _corpus(question: QuestionD0) -> tuple[Document, ...]:
    if question is QUESTION_MYOSITE:
        return (CollecteurPubMed(client_fige(FIXTURES / "http" / "gold"), pause_s=0).notice(PMID_MYOSITE)[0],)
    if question is QUESTION_THS:
        collecteur_ths = CollecteurPubMed(client_fige(FIXTURES / "http" / "ths"), pause_s=0)
        return tuple(collecteur_ths.notice(pmid)[0] for pmid in PMIDS_THS)
    collecteur = CollecteurPubMed(client_fige(FIXTURES / "http" / "vitamine_d"), pause_s=0)
    return tuple(collecteur.notice(pmid)[0] for pmid in PMIDS)


QUESTIONS = (QUESTION, QUESTION_MYOSITE, QUESTION_THS)


def _pieges(question: QuestionD0) -> tuple[Piege, ...]:
    return tuple(p for p in PIEGES if p.question is question)


def _claims(pieges: tuple[Piege, ...]) -> tuple[Claim, ...]:
    return tuple(Claim.extrait(f"c{i:03d}", p.source_id, p.citation) for i, p in enumerate(pieges, start=1))


def _deps(tmp_path: Path, question: QuestionD0) -> cycle_d0.Dependances:
    dossier = tmp_path / question.question_id
    return cycle_d0.Dependances(
        client=ClientLLM(config.mode_cassette(), config.dossier_cassettes()),
        ledger=RunLedger(f"red-team-{question.question_id}", dossier),
        collecteurs={},
        dossier_run=dossier,
    )


def _etat_filtre(tmp_path: Path, question: QuestionD0) -> tuple[EtatD0, cycle_d0.Dependances]:
    deps = _deps(tmp_path, question)
    etat = cycle_d0.demarrer(question, f"red-team-{question.question_id}", deps, _corpus(question))
    etat = cycle_d0._maj(etat, deps, claims=_claims(_pieges(question)))
    return cycle_d0.filtrer(etat, deps), deps


def _par_nom(etat: EtatD0, question: QuestionD0) -> dict[str, Claim]:
    return dict(zip((p.nom for p in _pieges(question)), etat.claims, strict=True))


# --------------------------------------------------------------------------- #
# Gates seules (0 LLM)
# --------------------------------------------------------------------------- #


def test_corpus_fige_reel() -> None:
    corpus = _corpus(QUESTION)
    assert [d.source_id for d in corpus] == [f"pmid:{p}" for p in PMIDS]
    assert all(d.textclean_version == config.TEXTCLEAN_VERSION for d in corpus)
    assert all("[TITRE]" in d.texte_nettoye for d in corpus)


def test_chaque_piege_a_une_question() -> None:
    assert sum(len(_pieges(q)) for q in QUESTIONS) == len(PIEGES)
    assert {p.nom for p in _pieges(QUESTION_MYOSITE)} == {"myosite_nct_sans_resultat", "myosite_futur"}


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.question_id)
def test_pieges_deterministes_aux_gates(tmp_path: Path, question: QuestionD0) -> None:
    etat, _ = _etat_filtre(tmp_path, question)
    par_nom = _par_nom(etat, question)
    for piege in _pieges(question):
        claim = par_nom[piege.nom]
        if piege.nom in SEMANTIQUES:
            assert claim.statut is S.EXTRACTED, (piege.nom, claim.motif_rejet)
            assert claim.drapeaux is not None
        else:
            assert claim.statut in piege.attendus, (piege.nom, claim.statut, claim.motif_rejet)
    for piege in _pieges(question):
        if piege.motif_exact is not None and piege.nom not in SEMANTIQUES:
            assert par_nom[piege.nom].motif_rejet == piege.motif_exact, piege.nom
        if piege.axes == TEMPORALITE:  # protocoles : le pré-filtre doit lever le drapeau
            drapeaux = par_nom[piege.nom].drapeaux
            assert drapeaux is not None and drapeaux.is_future_or_protocol, piege.nom


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.question_id)
def test_verdict_refus_sans_equation(tmp_path: Path, question: QuestionD0) -> None:
    """Gate mécanique du routeur : aucune équation candidate -> refus sans appel LLM."""
    etat, deps = _etat_filtre(tmp_path, question)
    etat = cycle_d0.router(etat, deps)
    assert etat.verdict is not None
    assert etat.verdict.verdict is Verdict.REFUS
    assert etat.verdict.motif_refus in (MotifRefus.CLAIMS_INSUFFISANTS, MotifRefus.PAS_D_EQUATION)
    assert deps.ledger.entrees == ()


# --------------------------------------------------------------------------- #
# Chaîne complète avec juges (cassettes)
# --------------------------------------------------------------------------- #


def test_red_team_complet(tmp_path: Path) -> None:
    echecs: dict[str, tuple[str, str | None]] = {}
    etats = {}
    for question in (QUESTION, QUESTION_MYOSITE):
        etat, deps = _etat_filtre(tmp_path, question)
        etat = cycle_d0.auditer(etat, deps)
        if question is QUESTION:
            etat = cycle_d0._maj(etat, deps, preuves=(FAUSSE_REFUTATION,))
            etat = cycle_d0.evaluer_preuves(etat, deps)
        etat = cycle_d0.router(etat, deps)
        etats[question.question_id] = (etat, deps)
        par_nom = _par_nom(etat, question)
        for p in _pieges(question):
            claim = par_nom[p.nom]
            ecart = None if claim.statut in p.attendus else f"statut {claim.statut.value}"
            echecs[p.nom] = (ecart or _ecarts_motif(p, claim) or "", claim.historique[-1].motif)
    echecs = {nom: e for nom, e in echecs.items() if e[0]}
    assert not echecs, echecs

    # Aucun échec de juge (sortie invalide, refus du fournisseur) ne doit passer pour un verdict.
    for etat, _ in etats.values():
        motifs = {t.motif for c in etat.claims for t in c.historique}
        motifs |= {t.motif for p in etat.preuves for t in p.historique} | {i.motif for i in etat.incidents}
        assert not motifs & MOTIFS_ECHEC_JUGE, motifs & MOTIFS_ECHEC_JUGE

    etat, _ = etats[QUESTION.question_id]
    (preuve,) = etat.preuves
    assert preuve.statut_preuve is StatutPreuve.REJECTED_SEMANTIC, preuve.historique[-1].motif
    assert preuve.historique[-1].porte is Porte.AUDITEUR
    assert axes_non_conformes(preuve.historique[-1].motif) & AXES_FAUSSE_REFUTATION, preuve.historique[-1].motif
    assert _par_nom(etat, QUESTION)["etabli_chapuy"].statut is S.ESTABLISHED  # la fausse réfutation ne contredit pas
    assert etat.verdict is not None and etat.verdict.motif_refus is MotifRefus.PAS_D_EQUATION

    etat_myosite, _ = etats[QUESTION_MYOSITE.question_id]
    assert etat_myosite.verdict is not None and etat_myosite.verdict.motif_refus is MotifRefus.CLAIMS_INSUFFISANTS
    if os.environ.get("METIS_CASSETTE") not in ("record", "record_missing"):
        assert all(e.cassette for _, deps in etats.values() for e in deps.ledger.entrees)


def test_vraie_refutation_confirmee(tmp_path: Path) -> None:
    """Échoue si aucune preuve CONFIRMED n'est produite sur la vraie réfutation propre.

    Le claim NHS est injecté en ESTABLISHED (porte simulée) : l'auditeur n'est appelé
    que sur la preuve adverse WHI.
    """
    etat, deps = _etat_filtre(tmp_path, QUESTION_THS)
    (claim,) = etat.claims
    assert claim.statut is S.EXTRACTED and claim.drapeaux is not None  # gates passées
    etat = cycle_d0._maj(etat, deps, claims=(injecter_etabli(claim),), preuves=(VRAIE_REFUTATION,))
    etat = cycle_d0.evaluer_preuves(etat, deps)
    etat = cycle_d0.router(etat, deps)

    (preuve,) = etat.preuves
    assert preuve.statut_preuve is StatutPreuve.CONFIRMED, f"aucune CONFIRMED : {preuve.historique[-1].motif}"
    assert axes_non_conformes(preuve.historique[-1].motif) == frozenset()
    (claim,) = etat.claims
    (piege,) = _pieges(QUESTION_THS)
    assert claim.statut in piege.attendus and _ecarts_motif(piege, claim) is None, claim.historique[-1]
    assert [t.motif for t in claim.historique[-2:]] == [MOTIF_INJECTION, "p001"]
    motifs = {t.motif for t in claim.historique} | {t.motif for t in preuve.historique}
    assert not motifs & MOTIFS_ECHEC_JUGE
    assert etat.verdict is not None and etat.verdict.motif_refus is MotifRefus.CLAIMS_INSUFFISANTS
    assert [e.role.value for e in deps.ledger.entrees] == ["auditeur"]  # la preuve seule
    if os.environ.get("METIS_CASSETTE") not in ("record", "record_missing"):
        assert all(e.cassette for e in deps.ledger.entrees)
