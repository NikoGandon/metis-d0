"""Étape 2 : gates documentaires textclean.py, verify.py, prefilter.py (0 LLM)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from metis import config
from metis.literature.prefilter import calculer_drapeaux, prefiltrer_claim, prefiltrer_preuve
from metis.literature.textclean import (
    construire_document,
    nettoyer_latex,
    nettoyer_pmc_xml,
    nettoyer_pubmed_xml,
    nettoyer_resume_arxiv,
)
from metis.literature.verify import (
    Ancrage,
    MotifVerify,
    normaliser,
    normaliser_citation,
    verifier_citation,
    verifier_claim,
    verifier_preuve,
)
from metis.schemas import (
    Claim,
    Document,
    EspeceCible,
    FormatSource,
    Grandeur,
    PreuveAdverse,
    QuestionD0,
    RequetesInitiales,
    StatutClaim,
    StatutPreuve,
)

FIXTURES = Path(__file__).parent / "fixtures" / "textclean"

PROSE = (
    "Rapamycin, an inhibitor of the mechanistic target of rapamycin complex 1, was administered "
    "orally to 120 participants aged 70 to 95 years. After 16 weeks, the response to influenza "
    "vaccination improved by roughly 20% compared with placebo, whereas adverse events were rare "
    "and mostly mild. These results suggest that mTORC1 inhibition may reverse aspects of "
    "immunosenescence in elderly humans."
)


def _doc(texte: str, source_id: str = "pmid:1", fmt: FormatSource = FormatSource.ARXIV_ABSTRACT) -> Document:
    return Document(
        source_id=source_id, format=fmt, texte_brut=texte, texte_nettoye=texte, textclean_version="test"
    )


def _corpus(*docs: Document) -> dict[str, Document]:
    return {d.source_id: d for d in docs}


def _ancrage(resultat: Ancrage | MotifVerify) -> Ancrage:
    assert isinstance(resultat, Ancrage), resultat
    return resultat


# --------------------------------------------------------------------------- #
# textclean.py
# --------------------------------------------------------------------------- #


def test_pmc_balises_et_structure() -> None:
    texte = nettoyer_pmc_xml((FIXTURES / "pmc_jats_minimal.xml").read_text(encoding="utf-8"))
    blocs = texte.split("\n\n")
    assert blocs[0] == "[TITRE] Rapamycin and in vivo ageing markers"  # <italic> aplati
    assert "Rapamycin extends lifespan in mice." in blocs  # <xref>[1]</xref> supprimé sans « mice . »
    assert (
        "Plasma levels reached 10−3 M and the H2O content was stable. Previous work [12] and reviews "
        "[3–5] agree, as summarised in Table 1." in blocs
    )  # sup/sub aplatis ; xref (Smith; Doe), [xref] et appels en exposant retirés ; [12] textuel et
    # renvoi au tableau conservés
    assert r"The rate law v = \frac{V_{\max} S}{K_m + S} held in all patients." in blocs  # tex-math
    assert "[ÉQUATION] I(S) = S_0 + I_0 - S" in blocs
    assert "[TABLEAU] Table 1 Baseline characteristics." in blocs
    assert "Group | n" in blocs and "Placebo | 12" in blocs
    assert "[NOTE DE TABLEAU] Values are means." in blocs
    assert "[LÉGENDE] Figure 1 Survival curves of treated patients." in blocs
    assert "Harrison" not in texte  # <back>/<ref-list> exclus


def test_pmc_xml_invalide_leve() -> None:
    with pytest.raises(Exception):
        nettoyer_pmc_xml("<article><body><p>non fermé</body></article>")
    with pytest.raises(ValueError, match="exactement un"):
        nettoyer_pmc_xml("<pmc-articleset><article/><article/></pmc-articleset>")


def test_pubmed_xml() -> None:
    xml = (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation><Article>"
        "<ArticleTitle>mTOR inhibition and <i>in vivo</i> immunity.</ArticleTitle>"
        '<Abstract><AbstractText Label="BACKGROUND">Ageing impairs immunity [1].</AbstractText>'
        '<AbstractText Label="RESULTS">Responses rose by 20%.</AbstractText></Abstract>'
        "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )
    assert nettoyer_pubmed_xml(xml).split("\n\n") == [
        "[TITRE] mTOR inhibition and in vivo immunity.",
        "[RÉSUMÉ]",
        "BACKGROUND: Ageing impairs immunity [1].",  # PubMed : rien n'est retiré
        "RESULTS: Responses rose by 20%.",
    ]


def _jats(paragraphe: str) -> str:
    return f"<article><body><p>{paragraphe}</p></body></article>"


@pytest.mark.parametrize(
    ("paragraphe", "attendu"),
    [
        ('lifespan in mice <xref ref-type="bibr" rid="b1">[1]</xref>.', "lifespan in mice."),
        ('lifespan [<xref ref-type="bibr" rid="b1">1</xref>].', "lifespan."),
        (
            'reviews [<xref ref-type="bibr" rid="b3">3</xref>–<xref ref-type="bibr" rid="b5">5</xref>] agree',
            "reviews agree",
        ),
        (
            'agree (<xref ref-type="bibr" rid="a">Smith 2010</xref>; <xref ref-type="bibr" rid="b">Doe</xref>), so',
            "agree, so",
        ),
        ('stable <sup><xref ref-type="bibr" rid="b2">2</xref>,<xref ref-type="fn" rid="n1">a</xref></sup>.', "stable."),
        ('see <xref ref-type="table" rid="t1">Table 1</xref> and <xref ref-type="fig" rid="f1">Fig. 2</xref>.',
         "see Table 1 and Fig. 2."),
        ('shown by <xref rid="b1">Harrison</xref> in mice', "shown by Harrison in mice"),  # sans ref-type : gardé
        ("Previous work [12] and reviews [3–5] agree.", "Previous work [12] and reviews [3–5] agree."),
    ],
)  # fmt: skip
def test_pmc_seuls_les_xref_bibr_fn_sont_retires(paragraphe: str, attendu: str) -> None:
    assert nettoyer_pmc_xml(_jats(paragraphe)) == attendu


CROCHETS = "on [0, 1], with [S] and [Ca2+] measured by [3H]thymidine uptake, see [12] and [3–5]."


def test_crochets_conserves_dans_tous_les_formats() -> None:
    assert nettoyer_pmc_xml(_jats("Defined " + CROCHETS)) == "Defined " + CROCHETS
    pubmed = (
        "<PubmedArticleSet><PubmedArticle><ArticleTitle>T</ArticleTitle><Abstract><AbstractText>"
        f"Defined {CROCHETS}</AbstractText></Abstract></PubmedArticle></PubmedArticleSet>"
    )
    assert nettoyer_pubmed_xml(pubmed).endswith("Defined " + CROCHETS)
    entree = (
        '<entry xmlns="http://www.w3.org/2005/Atom"><title>T</title>'
        f"<summary>Defined   {CROCHETS}</summary></entry>"
    )
    assert nettoyer_resume_arxiv(entree) == "[TITRE] T\n\n[RÉSUMÉ] Defined " + CROCHETS
    assert nettoyer_latex("Defined " + CROCHETS + " % fin") == "Defined " + CROCHETS + " "


def test_document_enregistre_la_version_textclean() -> None:
    entree = '<entry xmlns="http://www.w3.org/2005/Atom"><title>T</title><summary>Un résumé.</summary></entry>'
    doc = construire_document("arxiv:2401.00001", FormatSource.ARXIV_ABSTRACT, entree)
    assert doc.textclean_version == config.TEXTCLEAN_VERSION


def test_latex_seuls_les_commentaires_sont_retires() -> None:
    source = (
        "% préambule commenté\n"
        "Growth follows~\\cite{laird1964} the law % note interne\n"
        "\\begin{equation}\n"
        "W(t) = W_0 \\exp\\left(\\frac{A_0}{\\alpha}\\left(1 - e^{-\\alpha t}\\right)\\right)\n"
        "\\end{equation}\n"
        "with 50\\% of cases \\citep[p.~3]{gompertz}.\n"
    )
    assert nettoyer_latex(source) == (
        "\n"
        "Growth follows~\\cite{laird1964} the law \n"
        "\\begin{equation}\n"
        "W(t) = W_0 \\exp\\left(\\frac{A_0}{\\alpha}\\left(1 - e^{-\\alpha t}\\right)\\right)\n"
        "\\end{equation}\n"
        "with 50\\% of cases \\citep[p.~3]{gompertz}.\n"
    )


def test_appel_textuel_conserve_donc_requis_dans_la_citation() -> None:
    doc = construire_document("pmid:1", FormatSource.PUBMED_XML, (
        "<PubmedArticleSet><PubmedArticle><ArticleTitle>T</ArticleTitle><Abstract><AbstractText>"
        "Sirolimus lowered the incidence of infections in elderly participants [12] over one year."
        "</AbstractText></Abstract></PubmedArticle></PubmedArticleSet>"
    ))  # fmt: skip
    avec = "lowered the incidence of infections in elderly participants [12] over one year"
    sans = "lowered the incidence of infections in elderly participants over one year"
    assert isinstance(verifier_citation("pmid:1", avec, _corpus(doc)), Ancrage)
    assert verifier_citation("pmid:1", sans, _corpus(doc)) is MotifVerify.CITATION_NON_VERBATIM


# --------------------------------------------------------------------------- #
# verify.py : propriétés
# --------------------------------------------------------------------------- #

SENTINELLE = "☃"
_TEXTE = st.text(alphabet=st.characters(blacklist_characters=SENTINELLE), min_size=40, max_size=400)


def _coupe_une_grappe(texte: str, i: int) -> bool:
    return i < len(texte) and unicodedata.category(texte[i]).startswith("M")


@st.composite
def _texte_et_sous_chaine(draw: st.DrawFn) -> tuple[str, int, int]:
    texte = draw(_TEXTE)
    debut = draw(st.integers(0, len(texte) - 1))
    fin = draw(st.integers(debut + 1, len(texte)))
    assume(not _coupe_une_grappe(texte, debut) and not _coupe_une_grappe(texte, fin))
    return texte, debut, fin


@given(_texte_et_sous_chaine())
def test_propriete_toute_sous_chaine_passe(cas: tuple[str, int, int]) -> None:
    texte, debut, fin = cas
    citation = texte[debut:fin]
    resultat = verifier_citation("pmid:1", citation, _corpus(_doc(texte)))
    if len(normaliser_citation(citation)) >= 40:
        ancrage = _ancrage(resultat)
        # original[offsets] renormalisé == citation normalisée
        assert normaliser_citation(texte[ancrage.offsets.debut : ancrage.offsets.fin]) == normaliser_citation(
            citation
        )
    else:
        assert resultat is MotifVerify.CITATION_TROP_COURTE


@given(_texte_et_sous_chaine(), st.data())
def test_propriete_modifier_un_caractere_echoue(cas: tuple[str, int, int], data: st.DataObject) -> None:
    texte, debut, fin = cas
    citation = texte[debut:fin]
    positions = [k for k, c in enumerate(citation) if not c.isspace()]
    assume(positions)
    k = data.draw(st.sampled_from(positions))
    modifiee = citation[:k] + SENTINELLE + citation[k + 1 :]
    assert not isinstance(verifier_citation("pmid:1", modifiee, _corpus(_doc(texte))), Ancrage)


def test_toute_substitution_de_lettre_echoue_sur_de_la_prose() -> None:
    corpus = _corpus(_doc(PROSE))
    citation = PROSE[100:220]
    assert isinstance(verifier_citation("pmid:1", citation, corpus), Ancrage)
    for k, c in enumerate(citation):
        if c.isspace():
            continue
        remplacant = "q" if c.casefold() != "q" else "z"
        modifiee = citation[:k] + remplacant + citation[k + 1 :]
        assert verifier_citation("pmid:1", modifiee, corpus) is MotifVerify.CITATION_NON_VERBATIM, k


# --------------------------------------------------------------------------- #
# verify.py : cas ciblés
# --------------------------------------------------------------------------- #


def test_id_hors_corpus_du_run() -> None:
    run_precedent = _corpus(_doc(PROSE, source_id="pmid:999"))
    run_courant = _corpus(_doc(PROSE, source_id="pmid:1"))
    citation = PROSE[:80]
    assert isinstance(verifier_citation("pmid:999", citation, run_precedent), Ancrage)
    assert verifier_citation("pmid:999", citation, run_courant) is MotifVerify.SOURCE_HORS_CORPUS


def test_seuil_de_40_caracteres() -> None:
    corpus = _corpus(_doc(PROSE))
    assert verifier_citation("pmid:1", PROSE[:39], corpus) is MotifVerify.CITATION_TROP_COURTE
    assert isinstance(verifier_citation("pmid:1", PROSE[:40], corpus), Ancrage)
    assert verifier_citation("pmid:1", "", corpus) is MotifVerify.CITATION_TROP_COURTE
    # 39 caractères utiles entourés d'espaces : la longueur se mesure après normalisation
    assert verifier_citation("pmid:1", "   " + PROSE[:39] + "\n\n", corpus) is MotifVerify.CITATION_TROP_COURTE


def test_guillemets_tirets_espaces_unicode() -> None:
    source = "Patients said \u201cthe dose\u2013response curve was flat\u201d and it\u2019s\u00a0reproducible\u2009now."
    citation = 'patients said "the dose-response curve was flat" and it\'s reproducible  now.'
    ancrage = _ancrage(verifier_citation("pmid:1", citation, _corpus(_doc(source))))
    assert source[ancrage.offsets.debut : ancrage.offsets.fin] == source


def test_ligature_et_casse() -> None:
    source = "The ef\ufb01cacy of SIROLIMUS was con\ufb01rmed in the elderly cohort."
    citation = "the efficacy of sirolimus was confirmed in the elderly cohort."
    assert isinstance(verifier_citation("pmid:1", citation, _corpus(_doc(source))), Ancrage)


def test_table_d_offsets_vers_le_texte_original() -> None:
    source = "Pr\u00e9ambule\u00a0\u00a0ﬁn. " + PROSE
    normalise = normaliser(source)
    assert normalise.texte.startswith("préambule fin. ")
    # l'espace fusionnée couvre les deux insécables ; « fi » vient de la ligature unique
    espace = normalise.texte.index(" ")
    assert source[normalise.debuts[espace] : normalise.fins[espace]] == "\u00a0\u00a0"
    f, i = normalise.texte.index("fin"), normalise.texte.index("fin") + 1
    assert normalise.debuts[f] == normalise.debuts[i] and source[normalise.debuts[f]] == "ﬁ"

    citation = PROSE[50:150]
    ancrage = _ancrage(verifier_citation("pmid:1", citation, _corpus(_doc(source))))
    assert source[ancrage.offsets.debut : ancrage.offsets.fin] == citation


def test_e_accent_compose_et_decompose() -> None:
    source = "Une r\u00e9ponse immunitaire am\u00e9lior\u00e9e a \u00e9t\u00e9 observ\u00e9e chez les patients."
    citation = unicodedata.normalize("NFD", source)
    assert isinstance(verifier_citation("pmid:1", citation, _corpus(_doc(source))), Ancrage)


def test_correspondance_au_milieu_d_une_ligature_refusee() -> None:
    source = "Immunosenescence markers decreased in all treated participants (ﬁnal visit)."
    citation = "Immunosenescence markers decreased in all treated participants (f"
    assert verifier_citation("pmid:1", citation, _corpus(_doc(source))) is MotifVerify.CITATION_NON_VERBATIM


def test_fenetre_de_contexte_250_et_bornes() -> None:
    long = "a" * 300 + " " + PROSE + " " + "z" * 300
    citation = PROSE[:60]
    ancrage = _ancrage(verifier_citation("pmid:1", citation, _corpus(_doc(long))))
    debut, fin = ancrage.offsets.debut, ancrage.offsets.fin
    assert ancrage.context_window == long[debut - 250 : fin + 250]
    assert len(ancrage.context_window) == (fin - debut) + 500

    court = _ancrage(verifier_citation("pmid:1", citation, _corpus(_doc(PROSE))))
    assert court.offsets.debut == 0
    assert court.context_window == PROSE[: court.offsets.fin + 250]


PHRASE = "Serum IGF-1 concentrations remained unchanged throughout the trial."


def test_occurrences_multiples_toutes_les_fenetres() -> None:
    source = f"{PHRASE} Discussion follows. {PHRASE}"
    ancrage = _ancrage(verifier_citation("pmid:1", PHRASE, _corpus(_doc(source))))
    assert ancrage.nombre_occurrences == 2
    assert [o.offsets.debut for o in ancrage.occurrences] == [0, len(PHRASE) + 21]
    assert ancrage.offsets.debut == 0  # première retenue pour le claim


def test_au_plus_trois_fenetres_transmises() -> None:
    source = " ".join([PHRASE] * 5)
    claim, ancrage = verifier_claim(Claim.extrait("c1", "pmid:1", PHRASE), _corpus(_doc(source)))
    assert isinstance(ancrage, Ancrage) and ancrage.nombre_occurrences == 5
    assert len(claim.fenetres) == 3 and claim.nombre_occurrences == 5
    assert claim.fenetres[0] == claim.context_window


def test_drapeau_futur_vu_dans_une_occurrence_secondaire() -> None:
    citation = "Participants received 1 mg of sirolimus daily for twelve consecutive weeks."
    source = citation + " " + "x" * 600 + " " + citation + " The primary endpoint will be assessed at week 24."
    claim, _ = verifier_claim(Claim.extrait("c1", "pmid:1", citation), _corpus(_doc(source)))
    assert "will be assessed" not in claim.fenetres[0] and "will be assessed" in claim.fenetres[1]
    filtre = prefiltrer_claim(claim, _question())
    assert filtre.drapeaux is not None and filtre.drapeaux.is_future_or_protocol is True


def test_citation_traversant_des_balises_pmc() -> None:
    doc = construire_document(
        "pmc:PMC1", FormatSource.PMC_XML, (FIXTURES / "pmc_jats_minimal.xml").read_text(encoding="utf-8")
    )
    citation = "the H2O content was stable. Previous work [12] and reviews [3–5] agree, as summarised in Table 1"
    ancrage = _ancrage(verifier_citation("pmc:PMC1", citation, _corpus(doc)))
    assert doc.texte_nettoye[ancrage.offsets.debut : ancrage.offsets.fin] == citation
    titre = "Rapamycin and in vivo ageing markers. Some invented continuation"
    assert verifier_citation("pmc:PMC1", titre, _corpus(doc)) is MotifVerify.CITATION_NON_VERBATIM


def test_citation_inventee_rejetee() -> None:
    citation = "Rapamycin doubled the median lifespan of elderly humans in this trial."
    assert verifier_citation("pmid:1", citation, _corpus(_doc(PROSE))) is MotifVerify.CITATION_NON_VERBATIM


def test_verifier_claim_et_preuve() -> None:
    corpus = _corpus(_doc(PROSE))
    accepte, _ = verifier_claim(Claim.extrait("c1", "pmid:1", PROSE[:80]), corpus)
    assert accepte.statut is StatutClaim.EXTRACTED
    assert accepte.offsets is not None and accepte.context_window is not None

    rejete, resultat = verifier_claim(Claim.extrait("c2", "pmid:404", PROSE[:80]), corpus)
    assert resultat is MotifVerify.SOURCE_HORS_CORPUS
    assert rejete.statut is StatutClaim.REJECTED_VERBATIM
    assert rejete.motif_rejet == "source_hors_corpus"
    assert rejete.historique[-1].porte.value == "verify"

    preuve = PreuveAdverse.candidate("p1", "c1", "sirolimus vaccine elderly", "pmid:1", "trop court")
    rejetee, _ = verifier_preuve(preuve, corpus)
    assert rejetee.statut_preuve is StatutPreuve.REJECTED_VERBATIM
    assert rejetee.motif_rejet == "citation_absente_ou_trop_courte"


# --------------------------------------------------------------------------- #
# prefilter.py
# --------------------------------------------------------------------------- #


def _question(espece: EspeceCible = EspeceCible.HUMAN_IN_VIVO) -> QuestionD0:
    return QuestionD0(
        question_id="rapa-1",
        sous_question="La rapamycine améliore-t-elle la réponse vaccinale chez l'humain âgé ?",
        espece_cible=espece,
        population="adultes de 65 ans et plus",
        intervention="inhibiteur de mTOR à faible dose contre placebo",
        mesurande=Grandeur(nom="titre d'anticorps", unite="ratio"),
        requetes_initiales=RequetesInitiales(pubmed=("rapamycin vaccine elderly",)),
    )


def _claim_verifie(citation: str, contexte: str | None = None) -> Claim:
    texte = contexte if contexte is not None else citation
    claim, resultat = verifier_claim(Claim.extrait("c1", "pmid:1", citation), _corpus(_doc(texte)))
    assert isinstance(resultat, Ancrage), resultat
    return claim


def test_lexiques_a_limites_de_mot() -> None:
    lexiques = (
        config.LEXIQUE_ESPECES_NON_HUMAINES
        + config.LEXIQUE_SYSTEMES_NON_IN_VIVO
        + config.LEXIQUE_MARQUEURS_HUMAINS
        + config.LEXIQUE_FUTUR_PROTOCOLE
    )
    for motif in lexiques:
        assert motif.startswith(r"\b") and motif.endswith(r"\b"), motif
        re.compile(motif)


@pytest.mark.parametrize("mot", ["rate", "ratio", "migration", "pirate", "separately"])
def test_faux_amis_ne_declenchent_rien(mot: str) -> None:
    drapeaux = calculer_drapeaux(f"The {mot} was {mot.upper()} and {mot.capitalize()}.", (mot,))
    assert drapeaux.species_hits == () and drapeaux.human_hits == ()
    assert drapeaux.is_future_or_protocol is False


def test_unlike_in_mice_in_patients_non_rejete() -> None:
    claim = _claim_verifie("Unlike in mice, in patients we found no change in lean body mass after treatment.")
    filtre = prefiltrer_claim(claim, _question())
    assert filtre.statut is StatutClaim.EXTRACTED
    assert filtre.drapeaux is not None
    assert filtre.drapeaux.species_hits == ("mice",)
    assert filtre.drapeaux.human_hits == ("patients",)


def test_nct_seul_pose_le_drapeau_sans_changer_le_statut() -> None:
    claim = _claim_verifie("This randomized trial was registered at ClinicalTrials.gov as NCT01234567.")
    filtre = prefiltrer_claim(claim, _question())
    assert filtre.statut is StatutClaim.EXTRACTED
    assert filtre.drapeaux is not None and filtre.drapeaux.is_future_or_protocol is True
    assert filtre.drapeaux.species_hits == () and filtre.drapeaux.human_hits == ()


def test_espece_exclue_sans_marqueur_humain_rejetee() -> None:
    claim = _claim_verifie("Rapamycin extended median lifespan by 14% in genetically heterogeneous mice.")
    filtre = prefiltrer_claim(claim, _question())
    assert filtre.statut is StatutClaim.REJECTED_PREFILTER
    assert filtre.motif_rejet == "espece_exclue_sans_marqueur_humain: mice"
    assert filtre.historique[-1].porte.value == "prefilter"


@pytest.mark.parametrize(
    "citation",
    [
        "Everolimus reduced senescence markers in primary cell cultures from donors.",
        "Treated marmosets showed no significant change in fasting glucose levels.",
        "Lifespan of non\u2011human primates was not affected by the dietary intervention.",
        "Proliferation decreased in\u00a0vitro at nanomolar concentrations of the drug.",
    ],
)
def test_systemes_et_especes_non_humains_rejetes(citation: str) -> None:
    assert prefiltrer_claim(_claim_verifie(citation), _question()).statut is StatutClaim.REJECTED_PREFILTER


@pytest.mark.parametrize(
    ("citation", "hits"),
    [
        ("Rapamycin improved grip strength in adult mice after a twelve week treatment course.", "mice"),
        ("Hepatic autophagy markers increased in elderly rats given intermittent rapamycin dosing.", "rats"),
        ("Tumour growth slowed in patient-derived xenografts in mice receiving everolimus daily.", "mice"),
        ("Tumour growth slowed in patient derived xenografts in mice receiving everolimus daily.", "mice"),
        ("Senescence decreased in human-derived cell lines exposed to rapamycin for 48 hours.", "cell lines"),
    ],
)
def test_marqueurs_humains_trompeurs_rejetes(citation: str, hits: str) -> None:
    filtre = prefiltrer_claim(_claim_verifie(citation), _question(EspeceCible.HUMAN_IN_VIVO))
    assert filtre.statut is StatutClaim.REJECTED_PREFILTER
    assert filtre.drapeaux is not None
    assert filtre.drapeaux.species_hits == (hits,) and filtre.drapeaux.human_hits == ()


def test_tiret_de_ponctuation_ne_masque_pas_le_marqueur() -> None:
    citation = "Unlike in mice, among patients – all aged over 65 – lean mass did not change."
    filtre = prefiltrer_claim(_claim_verifie(citation), _question())
    assert filtre.statut is StatutClaim.EXTRACTED
    assert filtre.drapeaux is not None and filtre.drapeaux.human_hits == ("patients",)


def test_lexique_humain_sans_adults_ni_elderly() -> None:
    assert calculer_drapeaux("Healthy adults and elderly volunteers were enrolled.", ()).human_hits == ("volunteers",)


def test_regle_restreinte_a_human_in_vivo() -> None:
    citation = "Rapamycin extended median lifespan by 14% in genetically heterogeneous mice."
    for espece in (EspeceCible.ANIMAL, EspeceCible.IN_VITRO, EspeceCible.ANY):
        filtre = prefiltrer_claim(_claim_verifie(citation), _question(espece))
        assert filtre.statut is StatutClaim.EXTRACTED
        assert filtre.drapeaux is not None and filtre.drapeaux.species_hits == ("mice",)


def test_espece_dans_la_fenetre_seulement_non_rejetee() -> None:
    citation = "In older adults, the vaccine response improved after six weeks of low-dose treatment."
    contexte = "Earlier work in mice motivated this study. " + citation
    filtre = prefiltrer_claim(_claim_verifie(citation, contexte), _question())
    assert filtre.statut is StatutClaim.EXTRACTED
    assert filtre.drapeaux is not None and filtre.drapeaux.species_hits == ()


def test_futur_dans_la_fenetre_pose_le_drapeau() -> None:
    citation = "Participants received 1 mg of sirolimus daily for twelve consecutive weeks."
    contexte = citation + " The primary endpoint will be assessed at week 24."
    filtre = prefiltrer_claim(_claim_verifie(citation, contexte), _question())
    assert filtre.statut is StatutClaim.EXTRACTED
    assert filtre.drapeaux is not None and filtre.drapeaux.is_future_or_protocol is True


def test_prefiltre_exige_verify() -> None:
    with pytest.raises(ValueError, match="verify"):
        prefiltrer_claim(Claim.extrait("c1", "pmid:1", PROSE[:80]), _question())


def test_preuve_adverse_meme_porte() -> None:
    citation = "In rats, chronic rapamycin exposure impaired glucose tolerance within four weeks."
    preuve = PreuveAdverse.candidate("p1", "c1", "rapamycin glucose intolerance", "pmid:1", citation)
    preuve, _ = verifier_preuve(preuve, _corpus(_doc(citation)))
    filtree = prefiltrer_preuve(preuve, _question())
    assert filtree.statut_preuve is StatutPreuve.REJECTED_PREFILTER
    assert filtree.motif_rejet == "espece_exclue_sans_marqueur_humain: rats"
