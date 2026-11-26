"""Modèles Pydantic de Mêtis et machines à états des Claim / PreuveAdverse.

Tous les modèles sont gelés (frozen=True, extra="forbid") et ne contiennent que
des conteneurs immuables (tuple, MappingProxyType) : un objet validé ne peut plus
changer. Une transition de statut produit un nouvel objet et ajoute une entrée
à son historique ; toute transition absente des tables lève InvalidTransitionError,
y compris à la relecture d'un JSON persisté.
"""

from __future__ import annotations

import hashlib
import keyword
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    StringConstraints,
    model_validator,
)

from metis.storage import json_canonique

# --------------------------------------------------------------------------- #
# Types de base
# --------------------------------------------------------------------------- #


class _Gele(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _non_vide(valeur: str) -> str:
    if not valeur.strip():
        raise ValueError("chaîne vide ou blanche")
    return valeur


TexteNonVide = Annotated[str, AfterValidator(_non_vide)]

# Identifiants utilisés dans des chemins (runs/, catalogue/) : pas de « . » ni « / ».
Identifiant = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]

FlottantFini = Annotated[float, Field(allow_inf_nan=False)]
FlottantPositif = Annotated[float, Field(ge=0, allow_inf_nan=False)]

_RE_IDENT_PY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifiant_python(nom: str) -> str:
    if not _RE_IDENT_PY.fullmatch(nom) or keyword.iskeyword(nom) or "__" in nom:
        raise ValueError(f"identifiant Python invalide : {nom!r}")
    return nom


IdentifiantPython = Annotated[str, AfterValidator(_identifiant_python)]


def _geler(valeur: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(valeur))


def _degeler(valeur: Mapping[str, float]) -> dict[str, float]:
    return dict(valeur)


MappingFlottants = Annotated[
    Mapping[str, FlottantFini],
    AfterValidator(_geler),
    PlainSerializer(_degeler, return_type=dict[str, float]),
]


class InvalidTransitionError(Exception):
    """Transition de statut absente de la table, ou historique incohérent."""


# --------------------------------------------------------------------------- #
# QuestionD0
# --------------------------------------------------------------------------- #


class EspeceCible(StrEnum):
    HUMAN_IN_VIVO = "human_in_vivo"
    ANIMAL = "animal"
    IN_VITRO = "in_vitro"
    ANY = "any"


class Grandeur(_Gele):
    """Nom + unité (mesurande d'une QuestionD0, observable d'un verdict)."""

    nom: TexteNonVide
    unite: TexteNonVide


class RequetesInitiales(_Gele):
    pubmed: tuple[TexteNonVide, ...] = ()
    arxiv: tuple[TexteNonVide, ...] = ()

    @model_validator(mode="after")
    def _au_moins_une(self) -> Self:
        if not self.pubmed and not self.arxiv:
            raise ValueError("au moins une requête initiale (PubMed ou arXiv)")
        return self


class QuestionD0(_Gele):
    question_id: Identifiant
    sous_question: TexteNonVide
    espece_cible: EspeceCible
    # Texte libre : âge, sexe, cadre de vie de la population visée.
    population: TexteNonVide
    # Texte libre : molécule, dose, rythme, cointerventions, comparateur.
    intervention: TexteNonVide
    mesurande: Grandeur
    exclusions: tuple[TexteNonVide, ...] = ()
    requetes_initiales: RequetesInitiales


# --------------------------------------------------------------------------- #
# Documents du corpus
# --------------------------------------------------------------------------- #


class FormatSource(StrEnum):
    PUBMED_XML = "pubmed_xml"  # efetch PubMed : titre + abstract
    PMC_XML = "pmc_xml"  # PMC Open Access : texte intégral JATS
    ARXIV_ABSTRACT = "arxiv_abstract"  # résumé de l'API arXiv (texte)
    ARXIV_LATEX = "arxiv_latex"  # source LaTeX


class Document(_Gele):
    """Document téléchargé dans un run. Les offsets des claims visent texte_nettoye."""

    source_id: TexteNonVide
    format: FormatSource
    texte_brut: str
    texte_nettoye: str
    textclean_version: TexteNonVide


# --------------------------------------------------------------------------- #
# Machines à états
# --------------------------------------------------------------------------- #


class Porte(StrEnum):
    """Étape du pipeline qui a déclenché une transition."""

    EXTRACTEUR = "extracteur"
    VERIFY = "verify"
    PREFILTER = "prefilter"
    AUDITEUR = "auditeur"
    CONTRADICTEUR = "contradicteur"


class StatutClaim(StrEnum):
    EXTRACTED = "EXTRACTED"
    REJECTED_VERBATIM = "REJECTED_VERBATIM"
    REJECTED_PREFILTER = "REJECTED_PREFILTER"
    REJECTED_SEMANTIC = "REJECTED_SEMANTIC"
    CLAIMED_UNPROVEN = "CLAIMED_UNPROVEN"
    ESTABLISHED = "ESTABLISHED"
    CONTRADICTED = "CONTRADICTED"


class StatutPreuve(StrEnum):
    CANDIDATE = "CANDIDATE"
    REJECTED_VERBATIM = "REJECTED_VERBATIM"
    REJECTED_PREFILTER = "REJECTED_PREFILTER"
    REJECTED_SEMANTIC = "REJECTED_SEMANTIC"
    CONFIRMED = "CONFIRMED"


S = TypeVar("S", StatutClaim, StatutPreuve)

# (statut de départ, statut d'arrivée) -> portes autorisées à la déclencher.
# Le départ None désigne la création de l'objet.
TableTransitions = Mapping[tuple[S | None, S], frozenset[Porte]]

TRANSITIONS_CLAIM: TableTransitions[StatutClaim] = MappingProxyType(
    {
        (None, StatutClaim.EXTRACTED): frozenset({Porte.EXTRACTEUR}),
        (StatutClaim.EXTRACTED, StatutClaim.REJECTED_VERBATIM): frozenset({Porte.VERIFY}),
        (StatutClaim.EXTRACTED, StatutClaim.REJECTED_PREFILTER): frozenset({Porte.PREFILTER}),
        (StatutClaim.EXTRACTED, StatutClaim.REJECTED_SEMANTIC): frozenset({Porte.AUDITEUR}),
        (StatutClaim.EXTRACTED, StatutClaim.CLAIMED_UNPROVEN): frozenset({Porte.AUDITEUR}),
        (StatutClaim.EXTRACTED, StatutClaim.ESTABLISHED): frozenset({Porte.AUDITEUR}),
        (StatutClaim.ESTABLISHED, StatutClaim.CONTRADICTED): frozenset({Porte.CONTRADICTEUR}),
    }
)

TRANSITIONS_PREUVE: TableTransitions[StatutPreuve] = MappingProxyType(
    {
        (None, StatutPreuve.CANDIDATE): frozenset({Porte.CONTRADICTEUR}),
        (StatutPreuve.CANDIDATE, StatutPreuve.REJECTED_VERBATIM): frozenset({Porte.VERIFY}),
        (StatutPreuve.CANDIDATE, StatutPreuve.REJECTED_PREFILTER): frozenset({Porte.PREFILTER}),
        (StatutPreuve.CANDIDATE, StatutPreuve.REJECTED_SEMANTIC): frozenset({Porte.AUDITEUR}),
        (StatutPreuve.CANDIDATE, StatutPreuve.CONFIRMED): frozenset({Porte.AUDITEUR}),
    }
)

STATUTS_REJET_CLAIM = frozenset(
    {StatutClaim.REJECTED_VERBATIM, StatutClaim.REJECTED_PREFILTER, StatutClaim.REJECTED_SEMANTIC}
)
STATUTS_REJET_PREUVE = frozenset(
    {StatutPreuve.REJECTED_VERBATIM, StatutPreuve.REJECTED_PREFILTER, StatutPreuve.REJECTED_SEMANTIC}
)
# Statuts atteints seulement après passage de verify.py puis prefilter.py.
_STATUTS_POST_PREFILTRE_CLAIM = frozenset(
    {
        StatutClaim.REJECTED_PREFILTER,
        StatutClaim.REJECTED_SEMANTIC,
        StatutClaim.CLAIMED_UNPROVEN,
        StatutClaim.ESTABLISHED,
        StatutClaim.CONTRADICTED,
    }
)
_STATUTS_POST_PREFILTRE_PREUVE = frozenset(
    {StatutPreuve.REJECTED_PREFILTER, StatutPreuve.REJECTED_SEMANTIC, StatutPreuve.CONFIRMED}
)


def verifier_transition(table: TableTransitions[S], de: S | None, vers: S, porte: Porte) -> None:
    portes = table.get((de, vers))
    if portes is None:
        raise InvalidTransitionError(f"transition interdite : {de} -> {vers}")
    if porte not in portes:
        raise InvalidTransitionError(
            f"la porte {porte} ne peut pas déclencher {de} -> {vers} "
            f"(autorisées : {sorted(portes)})"
        )


class Transition(_Gele, Generic[S]):
    de: S | None
    vers: S
    porte: Porte
    motif: str | None = None


def _verifier_historique(
    table: TableTransitions[S], historique: tuple[Transition[S], ...], statut: S
) -> None:
    if not historique:
        raise InvalidTransitionError("historique vide")
    precedent: S | None = None
    for transition in historique:
        if transition.de != precedent:
            raise InvalidTransitionError(
                f"historique discontinu : {precedent} puis {transition.de} -> {transition.vers}"
            )
        verifier_transition(table, transition.de, transition.vers, transition.porte)
        precedent = transition.vers
    if precedent != statut:
        raise InvalidTransitionError(f"statut {statut} ≠ dernière transition {precedent}")


# --------------------------------------------------------------------------- #
# Claim et PreuveAdverse
# --------------------------------------------------------------------------- #


class Offsets(_Gele):
    """Bornes [debut, fin) dans le texte original (non normalisé) du document."""

    debut: Annotated[int, Field(ge=0)]
    fin: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _ordonne(self) -> Self:
        if self.fin <= self.debut:
            raise ValueError("offsets : fin doit être > debut")
        return self


class Occurrence(_Gele):
    """Une position de la citation dans texte_nettoye et sa fenêtre de contexte."""

    offsets: Offsets
    context_window: str


MAX_OCCURRENCES_CONSERVEES = 3


class Drapeaux(_Gele):
    is_future_or_protocol: bool
    species_hits: tuple[str, ...] = ()
    human_hits: tuple[str, ...] = ()


class _Citation(_Gele):
    """Champs communs aux objets cités (Claim, PreuveAdverse).

    source_id et citation_verbatim ne sont volontairement pas contraints : une
    citation vide, courte ou hors corpus doit finir en REJECTED_VERBATIM par la
    gate verify.py, pas en erreur de parsing.
    """

    source_id: str
    citation_verbatim: str
    # Première occurrence (retenue) ; les suivantes, au plus 2, partent aussi à l'auditeur.
    offsets: Offsets | None = None
    context_window: str | None = None
    occurrences_supplementaires: Annotated[
        tuple[Occurrence, ...], Field(max_length=MAX_OCCURRENCES_CONSERVEES - 1)
    ] = ()
    nombre_occurrences: Annotated[int, Field(ge=0)] = 0  # total trouvé par verify.py
    drapeaux: Drapeaux | None = None

    @property
    def fenetres(self) -> tuple[str, ...]:
        """Toutes les fenêtres conservées, première occurrence en tête."""
        if self.context_window is None:
            return ()
        return (self.context_window, *(o.context_window for o in self.occurrences_supplementaires))

    def _verifier_ancrage(self, post_prefiltre: bool) -> None:
        if (self.offsets is None) != (self.context_window is None):
            raise ValueError("offsets et context_window sont posés ensemble par verify.py")
        if self.offsets is None and (self.occurrences_supplementaires or self.nombre_occurrences):
            raise ValueError("occurrences sans ancrage verify.py")
        conservees = 1 + len(self.occurrences_supplementaires)
        if self.offsets is not None and conservees != min(self.nombre_occurrences, MAX_OCCURRENCES_CONSERVEES):
            raise ValueError("occurrences conservées incohérentes avec nombre_occurrences")
        if self.drapeaux is not None and self.offsets is None:
            raise ValueError("drapeaux posés sans ancrage verify.py")
        if post_prefiltre and self.drapeaux is None:
            raise ValueError("ce statut exige le passage de verify.py et prefilter.py")

    def avec_ancrage(self, occurrences: tuple[Occurrence, ...], nombre_occurrences: int) -> Self:
        """Résultat de verify.py (sans changement de statut). occurrences[0] est retenue."""
        if self.offsets is not None:
            raise InvalidTransitionError("ancrage déjà posé")
        if not occurrences:
            raise ValueError("au moins une occurrence")
        premiere, *suivantes = occurrences
        return self.model_validate(
            {
                **dict(self),
                "offsets": premiere.offsets,
                "context_window": premiere.context_window,
                "occurrences_supplementaires": tuple(suivantes),
                "nombre_occurrences": nombre_occurrences,
            }
        )

    def avec_drapeaux(self, drapeaux: Drapeaux) -> Self:
        """Résultat de prefilter.py (sans changement de statut)."""
        if self.drapeaux is not None:
            raise InvalidTransitionError("drapeaux déjà posés")
        return self.model_validate({**dict(self), "drapeaux": drapeaux})


class Claim(_Citation):
    claim_id: Identifiant
    statut: StatutClaim
    historique: tuple[Transition[StatutClaim], ...]
    motif_rejet: str | None = None

    @classmethod
    def extrait(cls, claim_id: str, source_id: str, citation_verbatim: str) -> Claim:
        return cls(
            claim_id=claim_id,
            source_id=source_id,
            citation_verbatim=citation_verbatim,
            statut=StatutClaim.EXTRACTED,
            historique=(Transition[StatutClaim](de=None, vers=StatutClaim.EXTRACTED, porte=Porte.EXTRACTEUR),),
        )

    @model_validator(mode="after")
    def _coherence(self) -> Self:
        _verifier_historique(TRANSITIONS_CLAIM, self.historique, self.statut)
        self._verifier_ancrage(self.statut in _STATUTS_POST_PREFILTRE_CLAIM)
        if self.statut in STATUTS_REJET_CLAIM and not (self.motif_rejet and self.motif_rejet.strip()):
            raise ValueError(f"{self.statut} exige un motif_rejet")
        if self.statut not in STATUTS_REJET_CLAIM and self.motif_rejet is not None:
            raise ValueError("motif_rejet réservé aux statuts de rejet")
        return self

    def transitionner(self, vers: StatutClaim, porte: Porte, motif: str | None = None) -> Claim:
        verifier_transition(TRANSITIONS_CLAIM, self.statut, vers, porte)
        transition = Transition[StatutClaim](de=self.statut, vers=vers, porte=porte, motif=motif)
        return Claim.model_validate(
            {
                **dict(self),
                "statut": vers,
                "historique": (*self.historique, transition),
                "motif_rejet": motif if vers in STATUTS_REJET_CLAIM else None,
            }
        )


class PreuveAdverse(_Citation):
    """Candidat de réfutation d'un Claim ESTABLISHED. Type distinct d'un Claim.

    Ajouts par rapport à la liste de champs de la spec : historique et motif_rejet,
    nécessaires au cimetière du rapport (porte qui l'a tué + motif).
    """

    preuve_id: Identifiant
    claim_cible_id: Identifiant
    requete: TexteNonVide
    statut_preuve: StatutPreuve
    historique: tuple[Transition[StatutPreuve], ...]
    motif_rejet: str | None = None

    @classmethod
    def candidate(
        cls, preuve_id: str, claim_cible_id: str, requete: str, source_id: str, citation_verbatim: str
    ) -> PreuveAdverse:
        return cls(
            preuve_id=preuve_id,
            claim_cible_id=claim_cible_id,
            requete=requete,
            source_id=source_id,
            citation_verbatim=citation_verbatim,
            statut_preuve=StatutPreuve.CANDIDATE,
            historique=(
                Transition[StatutPreuve](de=None, vers=StatutPreuve.CANDIDATE, porte=Porte.CONTRADICTEUR),
            ),
        )

    @model_validator(mode="after")
    def _coherence(self) -> Self:
        _verifier_historique(TRANSITIONS_PREUVE, self.historique, self.statut_preuve)
        self._verifier_ancrage(self.statut_preuve in _STATUTS_POST_PREFILTRE_PREUVE)
        rejet = self.statut_preuve in STATUTS_REJET_PREUVE
        if rejet and not (self.motif_rejet and self.motif_rejet.strip()):
            raise ValueError(f"{self.statut_preuve} exige un motif_rejet")
        if not rejet and self.motif_rejet is not None:
            raise ValueError("motif_rejet réservé aux statuts de rejet")
        return self

    def transitionner(self, vers: StatutPreuve, porte: Porte, motif: str | None = None) -> PreuveAdverse:
        verifier_transition(TRANSITIONS_PREUVE, self.statut_preuve, vers, porte)
        transition = Transition[StatutPreuve](de=self.statut_preuve, vers=vers, porte=porte, motif=motif)
        return PreuveAdverse.model_validate(
            {
                **dict(self),
                "statut_preuve": vers,
                "historique": (*self.historique, transition),
                "motif_rejet": motif if vers in STATUTS_REJET_PREUVE else None,
            }
        )


# --------------------------------------------------------------------------- #
# VerdictRoutage
# --------------------------------------------------------------------------- #


class Verdict(StrEnum):
    SIMULABLE = "simulable"
    REFUS = "refus"


class MotifRefus(StrEnum):
    PAS_D_EQUATION = "pas_d_equation"
    EQUATION_NON_VERBATIM = "equation_non_verbatim"
    PAS_D_OBSERVABLE = "pas_d_observable"
    QUESTION_DOCUMENTAIRE = "question_documentaire"
    CLAIMS_INSUFFISANTS = "claims_insuffisants"
    PARAMETRES_INACCESSIBLES = "parametres_inaccessibles"
    JUGE_REFUSE = "juge_refuse"


class VerdictRoutage(_Gele):
    verdict: Verdict
    motif_refus: MotifRefus | None = None
    papier_cible: str | None = None
    citation_equation: str | None = None
    observable: Grandeur | None = None

    @model_validator(mode="after")
    def _coherence(self) -> Self:
        if self.verdict is Verdict.SIMULABLE:
            if self.motif_refus is not None:
                raise ValueError("simulable sans motif_refus")
            if not self.papier_cible or not self.citation_equation or self.observable is None:
                raise ValueError("simulable exige papier_cible, citation_equation et observable")
        else:
            if self.motif_refus is None:
                raise ValueError("refus exige un motif_refus")
            if self.papier_cible is not None:
                raise ValueError("papier_cible doit être None en cas de refus")
        return self


# --------------------------------------------------------------------------- #
# Pipeline D0 : journaux du run
# --------------------------------------------------------------------------- #


class BaseLitterature(StrEnum):
    PUBMED = "pubmed"  # PubMed (abstracts) + PMC Open Access (texte intégral)
    ARXIV = "arxiv"


class OrigineRequete(StrEnum):
    INITIALE = "initiale"
    ADVERSE = "adverse"


class RequeteJournal(_Gele):
    """Requête exécutée par un collecteur, y compris sans résultat ou en erreur."""

    origine: OrigineRequete
    base: BaseLitterature
    texte: str
    claim_cible_id: str | None = None
    source_ids: tuple[str, ...] = ()
    erreur: str | None = None

    @property
    def nb_resultats(self) -> int:
        return len(self.source_ids)


class CandidatEquation(_Gele):
    """Équation proposée par l'extracteur ; jugée d'abord par la gate du routeur."""

    source_id: str
    citation_equation: str
    observable_nom: str
    observable_unite: str


class Incident(_Gele):
    """Échec journalisé d'une étape (sortie LLM invalide, collecte en erreur…)."""

    etape: str
    cible: str
    motif: str


class EtatD0(_Gele):
    """État complet d'un run D0, persisté après chaque étape dans runs/<run_id>/."""

    run_id: Identifiant
    question: QuestionD0
    documents: tuple[Document, ...] = ()
    requetes: tuple[RequeteJournal, ...] = ()
    claims: tuple[Claim, ...] = ()
    preuves: tuple[PreuveAdverse, ...] = ()
    candidats_equation: tuple[CandidatEquation, ...] = ()
    incidents: tuple[Incident, ...] = ()
    verdict: VerdictRoutage | None = None

    @property
    def corpus(self) -> Mapping[str, Document]:
        return MappingProxyType({d.source_id: d for d in self.documents})


# --------------------------------------------------------------------------- #
# Sorties LLM (schémas transmis aux modèles)
# --------------------------------------------------------------------------- #
# Volontairement sans contraintes numériques ni de longueur : les limites sont
# appliquées par le harnais après validation, pour rester dans le sous-ensemble de
# JSON Schema accepté par les sorties structurées.


class ClaimPropose(_Gele):
    source_id: str
    citation_verbatim: str


class EquationProposee(_Gele):
    source_id: str
    citation_equation: str
    observable_nom: str
    observable_unite: str


class SortieExtracteur(_Gele):
    claims: tuple[ClaimPropose, ...]
    equations: tuple[EquationProposee, ...]


class AxeJugement(_Gele):
    verdict: Literal["conforme", "non_conforme", "indetermine"]
    justification: str


class EvaluationFenetre(_Gele):
    indice_contexte: int
    espece_et_modele: AxeJugement
    temporalite_et_statut_de_preuve: AxeJugement
    mesurande: AxeJugement
    population: AxeJugement
    intervention: AxeJugement
    statut: Literal["REJECTED_SEMANTIC", "CLAIMED_UNPROVEN", "ESTABLISHED"]
    motif: str


class SortieAuditeur(_Gele):
    evaluations: tuple[EvaluationFenetre, ...]


class EvaluationFenetrePreuve(_Gele):
    indice_contexte: int
    espece_et_modele: AxeJugement
    temporalite_et_statut_de_preuve: AxeJugement
    mesurande: AxeJugement
    population: AxeJugement
    intervention: AxeJugement
    force_de_refutation: AxeJugement
    statut: Literal["REJECTED_SEMANTIC", "CONFIRMED"]
    motif: str


class SortieAuditeurPreuve(_Gele):
    evaluations: tuple[EvaluationFenetrePreuve, ...]


class RequeteAdverse(_Gele):
    base: Literal["pubmed", "arxiv"]
    texte: str


class SortieRequetesAdverses(_Gele):
    requetes: tuple[RequeteAdverse, ...]


class PreuveProposee(_Gele):
    source_id: str
    citation_verbatim: str


class SortiePreuves(_Gele):
    preuves: tuple[PreuveProposee, ...]


class SortieRouteur(_Gele):
    decision: Literal["confirmer", "refuser"]
    motif_refus: Literal[
        "pas_d_equation", "question_documentaire", "parametres_inaccessibles", "juge_refuse"
    ] | None
    justification: str


# --------------------------------------------------------------------------- #
# CommandeD1
# --------------------------------------------------------------------------- #


class OrigineContrat(StrEnum):
    MANUEL = "manuel"  # contrats de référence : citation non revérifiée
    PONT_D0 = "pont_d0"  # citation revérifiée contre le corpus du run D0


class SignatureFonction(_Gele):
    """Signature typée d'un outil forgé. Toutes les entrées et sorties sont des float.

    Les sorties sont nommées pour que les invariants et les cas puissent les viser.
    """

    nom: IdentifiantPython
    entrees: Annotated[tuple[IdentifiantPython, ...], Field(min_length=1)]
    sorties: Annotated[tuple[IdentifiantPython, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _noms_distincts(self) -> Self:
        noms = (self.nom, *self.entrees, *self.sorties)
        if len(set(noms)) != len(noms):
            raise ValueError("nom, entrées et sorties doivent être deux à deux distincts")
        return self

    def rendre(self) -> str:
        arguments = ", ".join(f"{e}: float" for e in self.entrees)
        n = len(self.sorties)
        retour = "float" if n == 1 else f"tuple[{', '.join(['float'] * n)}]"
        return f"{self.nom}({arguments}) -> {retour}"


class CasTest(_Gele):
    entrees: MappingFlottants
    attendu: MappingFlottants


class Intervalle(_Gele):
    bas: FlottantFini
    haut: FlottantFini

    @model_validator(mode="after")
    def _ordonne(self) -> Self:
        if self.haut < self.bas:
            raise ValueError("intervalle : haut < bas")
        return self


def _geler_domaine(valeur: Mapping[str, Intervalle]) -> Mapping[str, Intervalle]:
    return MappingProxyType(dict(valeur))


def _degeler_domaine(valeur: Mapping[str, Intervalle]) -> dict[str, Intervalle]:
    return dict(valeur)


Domaine = Annotated[
    Mapping[str, Intervalle],
    AfterValidator(_geler_domaine),
    PlainSerializer(_degeler_domaine, return_type=dict[str, Intervalle]),
]


class ContrainteRatioMax(_Gele):
    """Contrainte d'échantillonnage de la grille : num / den ≤ max (tirage rejeté sinon).

    Type fermé et déclaratif : num et den sont des noms d'entrées, jamais une
    expression évaluée. Seul type de contrainte existant ; en ajouter un exige un
    nouveau modèle ici et son interprétation dans check_outil.py.
    """

    type: Literal["ratio_max"]
    num: str
    den: str
    max: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class InvariantBounded(_Gele):
    type: Literal["bounded"]
    sortie: str
    # Un nombre, None (pas de borne) ou le nom d'une entrée de la signature.
    min: FlottantFini | str | None
    max: FlottantFini | str | None

    @model_validator(mode="after")
    def _une_borne(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("bounded exige au moins une borne")
        return self


class InvariantConservationSum(_Gele):
    type: Literal["conservation_sum"]
    sorties: Annotated[tuple[str, ...], Field(min_length=2)]
    cible: FlottantFini
    tol: FlottantPositif


class InvariantMonotonic(_Gele):
    type: Literal["monotonic"]
    sortie: str
    entree: str
    direction: Literal["increasing", "decreasing"]


Invariant = Annotated[
    InvariantBounded | InvariantConservationSum | InvariantMonotonic,
    Field(discriminator="type"),
]


class CommandeD1(_Gele):
    commande_id: Identifiant
    piste_id: Identifiant
    sujet_id: Identifiant
    origine: OrigineContrat
    run_id_d0: Identifiant | None = None
    source_id: TexteNonVide
    citation_equation: TexteNonVide
    convention: TexteNonVide
    signature_fonction: SignatureFonction
    description_pour_forgeron: TexteNonVide
    parametres: MappingFlottants
    epsilon_abs: FlottantPositif
    epsilon_rel: FlottantPositif
    cas_publics: Annotated[tuple[CasTest, ...], Field(min_length=1)]
    cas_caches: Annotated[tuple[CasTest, ...], Field(min_length=1)]
    graine_grille: Annotated[int, Field(ge=0)]
    domaine_grille: Domaine
    contraintes_grille: tuple[ContrainteRatioMax, ...] = ()
    invariants: tuple[Invariant, ...] = ()

    @model_validator(mode="after")
    def _coherence(self) -> Self:
        if (self.origine is OrigineContrat.PONT_D0) != (self.run_id_d0 is not None):
            raise ValueError("run_id_d0 est requis pour origine=pont_d0 et interdit sinon")
        entrees = set(self.signature_fonction.entrees)
        sorties = set(self.signature_fonction.sorties)
        for i, cas in enumerate((*self.cas_publics, *self.cas_caches)):
            if set(cas.entrees) != entrees or set(cas.attendu) != sorties:
                raise ValueError(f"cas n°{i} : entrées/sorties ≠ signature")
        if set(self.domaine_grille) != entrees:
            raise ValueError("domaine_grille doit couvrir exactement les entrées")
        for contrainte in self.contraintes_grille:
            if {contrainte.num, contrainte.den} - entrees:
                raise ValueError("contrainte_grille : entrée absente de la signature")
            if self.domaine_grille[contrainte.den].bas <= 0:
                raise ValueError("contrainte_grille : dénominateur non strictement positif")
        for inv in self.invariants:
            _verifier_invariant(inv, entrees, sorties)
        return self

    @property
    def contrat_hash(self) -> str:
        return hashlib.sha256(json_canonique(self.model_dump(mode="json"))).hexdigest()


def _verifier_invariant(
    inv: InvariantBounded | InvariantConservationSum | InvariantMonotonic,
    entrees: set[str],
    sorties: set[str],
) -> None:
    if isinstance(inv, InvariantBounded):
        if inv.sortie not in sorties:
            raise ValueError(f"bounded : sortie inconnue {inv.sortie!r}")
        for borne in (inv.min, inv.max):
            if isinstance(borne, str) and borne not in entrees:
                raise ValueError(f"bounded : borne {borne!r} n'est pas une entrée")
    elif isinstance(inv, InvariantConservationSum):
        if len(set(inv.sorties)) != len(inv.sorties) or set(inv.sorties) - sorties:
            raise ValueError("conservation_sum : sorties inconnues ou dupliquées")
    else:
        if inv.sortie not in sorties or inv.entree not in entrees:
            raise ValueError("monotonic : sortie ou entrée inconnue")


# --------------------------------------------------------------------------- #
# LLM : messages, ledger, cassettes
# --------------------------------------------------------------------------- #


class RoleLLM(StrEnum):
    EXTRACTEUR = "extracteur"
    AUDITEUR = "auditeur"
    CONTRADICTEUR = "contradicteur"
    ROUTEUR = "routeur"
    FORGERON = "forgeron"


class Message(_Gele):
    role: Literal["system", "user", "assistant"]
    content: str


class EntreeLedger(_Gele):
    role: RoleLLM
    model: TexteNonVide
    prompt_tokens: Annotated[int, Field(ge=0)]
    completion_tokens: Annotated[int, Field(ge=0)]
    cost_usd: FlottantPositif
    latency_ms: FlottantPositif
    cassette: bool
    horodatage: datetime
    fichier_reponse: str | None = None  # réponse brute archivée, relatif au dossier du run
    refus_fournisseur: bool = False  # finish_reason content_filter / native_finish_reason refusal
    fournisseur: str | None = None  # fournisseur effectif renvoyé par OpenRouter (champ provider)


class CoutsRun(_Gele):
    """Contenu de runs/<run_id>/couts.json."""

    run_id: Identifiant
    entrees: tuple[EntreeLedger, ...]
    cout_total_usd: FlottantPositif


class Cassette(_Gele):
    """Réponse LLM réelle enregistrée, clée par sha256(modèle + prompt + schéma)."""

    cle: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    modele: TexteNonVide
    date_enregistrement: datetime
    nom_schema: TexteNonVide
    messages: tuple[Message, ...]
    latency_ms: FlottantPositif
    reponse_brute: JsonValue
    fournisseur: str | None = None  # fournisseur effectif renvoyé par OpenRouter (champ provider)
