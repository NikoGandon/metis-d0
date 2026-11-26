"""Configuration centrale de Mêtis.

Seul fichier du dépôt où apparaissent des noms de modèles. Les valeurs lues dans
l'environnement le sont à l'appel (et non à l'import) pour que les tests puissent
les surcharger avec monkeypatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from metis.schemas import RoleLLM

# --------------------------------------------------------------------------- #
# Modèles et prix
# --------------------------------------------------------------------------- #

ENV_MODELE_JUGE = "METIS_MODEL_JUDGE"
ENV_MODELE_OUVRIER = "METIS_MODEL_WORKER"

MODELE_JUGE_DEFAUT = "anthropic/claude-opus-5.5"
MODELE_OUVRIER_DEFAUT = "anthropic/claude-haiku-5.5"


@dataclass(frozen=True)
class Tarif:
    """Prix en USD par million de tokens."""

    entree_usd_par_mtok: float
    sortie_usd_par_mtok: float


# Identifiants et tarifs vérifiés le 2026-11-03 sur le catalogue OpenRouter
# (GET https://openrouter.ai/api/v1/models, champs id et pricing) ; identiques aux
# tarifs publics Anthropic. À revérifier à chaque changement de modèle.
# Haiku 5.5 : tarif Anthropic valable jusqu'à 100K tokens d'entrée par requête
# (au-delà 0,50 / 2,50) ; aucun prompt Mêtis n'approche ce seuil.
TABLE_PRIX: dict[str, Tarif] = {
    "anthropic/claude-opus-5.5": Tarif(4.00, 20.00),
    "anthropic/claude-sonnet-5.5": Tarif(2.00, 10.00),
    "anthropic/claude-haiku-5.5": Tarif(0.10, 0.50),
    # Juges comparés par evals/run_evals.py (catalogue OpenRouter du 2026-11-04).
    # « Qwen 3.8 Max » = qwen3.8-max-0902 ; la variante « Prime » (4 / 12) est le même
    # modèle servi à plus haut débit.
    "qwen/qwen3.8-max-0902": Tarif(2.00, 6.00),
    "moonshotai/kimi-k3.1-pro": Tarif(2.00, 10.00),
}

# Juges comparés par défaut dans evals/run_evals.py.
JUGES_COMPARES = (
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
    "qwen/qwen3.8-max-0902",
    "moonshotai/kimi-k3.1-pro",
)

# Rôles servis par le modèle ouvrier ; tous les autres sont des juges.
ROLES_OUVRIER = frozenset({RoleLLM.FORGERON})


def modele_juge() -> str:
    return os.environ.get(ENV_MODELE_JUGE, MODELE_JUGE_DEFAUT)


def modele_ouvrier() -> str:
    return os.environ.get(ENV_MODELE_OUVRIER, MODELE_OUVRIER_DEFAUT)


# Modèle par rôle de juge ; à défaut, METIS_MODEL_JUDGE. Le forgeron suit METIS_MODEL_WORKER.
ENV_MODELE_PAR_ROLE = {
    RoleLLM.EXTRACTEUR: "METIS_MODEL_EXTRACTEUR",
    RoleLLM.AUDITEUR: "METIS_MODEL_AUDITEUR",
    RoleLLM.CONTRADICTEUR: "METIS_MODEL_CONTRADICTEUR",
    RoleLLM.ROUTEUR: "METIS_MODEL_ROUTEUR",
}


def modele_pour_role(role: RoleLLM) -> str:
    if role in ROLES_OUVRIER:
        return modele_ouvrier()
    return os.environ.get(ENV_MODELE_PAR_ROLE[role]) or modele_juge()


# Épinglage optionnel du fournisseur OpenRouter par rôle : METIS_PROVIDER_<ROLE> =
# liste séparée par des virgules (ex. « anthropic »), envoyée comme
# provider.order avec allow_fallbacks=false. Non défini : routage OpenRouter par défaut.
def fournisseurs_pour_role(role: RoleLLM) -> tuple[str, ...] | None:
    valeur = os.environ.get(f"METIS_PROVIDER_{role.value.upper()}", "")
    fournisseurs = tuple(f.strip() for f in valeur.split(",") if f.strip())
    return fournisseurs or None


# --------------------------------------------------------------------------- #
# Client LLM (OpenRouter) et cassettes
# --------------------------------------------------------------------------- #

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_LLM_BASE_URL = "METIS_LLM_BASE_URL"
ENV_OPENROUTER_CLE = "OPENROUTER_API_KEY"
TIMEOUT_LLM_S = 180.0

PREFIXES_MODELES_LOCAUX: tuple[str, ...] = ("ollama/", "local/")
HOTES_LOCAUX: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


def llm_base_url() -> str:
    return os.environ.get(ENV_LLM_BASE_URL) or OPENROUTER_URL


def est_modele_local(modele: str) -> bool:
    return modele.startswith(PREFIXES_MODELES_LOCAUX)


def nom_modele_distant(modele: str) -> str:
    for prefixe in PREFIXES_MODELES_LOCAUX:
        if modele.startswith(prefixe):
            return modele.removeprefix(prefixe)
    return modele

# Plafond de tokens de sortie par rôle (max_tokens de la requête). Il borne aussi la
# réservation du budget et celle qu'OpenRouter fait sur le solde. Hors clé de cassette :
# le modifier n'invalide aucune cassette. Maxima observés au 2026-11-04 (cassettes) :
# auditeur 926 (34 appels, preuves 631), extracteur 565 (3 appels) ; aucun appel
# observé pour le contradicteur ni le routeur, plafonds dimensionnés sur leurs schémas.
# Une sortie tronquée (finish_reason=length) donne un JSON invalide : réparation, puis
# fail-closed juge_sortie_invalide.
MAX_TOKENS_PAR_ROLE = {
    RoleLLM.AUDITEUR: 2_000,  # 6 axes × au plus 3 contextes ; ~2,2 × le maximum observé
    RoleLLM.EXTRACTEUR: 3_000,  # jusqu'à 8 claims et 3 équations cités verbatim ; ~5 × le max observé
    RoleLLM.CONTRADICTEUR: 1_500,  # 3 requêtes, ou 3 passages verbatim par segment
    RoleLLM.ROUTEUR: 1_000,  # décision, motif et justification
    RoleLLM.FORGERON: 4_000,  # corps de fonction de 8 Ko au plus
}


def max_tokens_pour_role(role: RoleLLM) -> int:
    return MAX_TOKENS_PAR_ROLE[role]


# Budget par run (RunLedger) : un appel n'est envoyé que si le coût déjà dépensé plus
# son coût maximal possible (borne : octets du prompt et du schéma comme tokens
# d'entrée, max_tokens du rôle en sortie) reste sous le budget. Rejeux non comptés.
ENV_BUDGET = "METIS_BUDGET_USD"
BUDGET_USD_DEFAUT = 10.0


def budget_usd() -> float:
    valeur = float(os.environ.get(ENV_BUDGET, BUDGET_USD_DEFAUT))
    if not valeur > 0 or valeur == float("inf"):
        raise ValueError(f"{ENV_BUDGET} doit être un nombre fini > 0")
    return valeur


class ModeCassette(StrEnum):
    LIVE = "live"  # réseau, aucune cassette lue ni écrite
    REPLAY = "replay"  # lecture seule ; clé absente -> CassetteMissError
    RECORD = "record"  # réseau, puis (ré)écriture de la cassette
    RECORD_MISSING = "record_missing"  # rejoue si la cassette existe, sinon réseau puis écriture


ENV_CASSETTE = "METIS_CASSETTE"
ENV_DOSSIER_CASSETTES = "METIS_CASSETTE_DIR"


def mode_cassette() -> ModeCassette:
    return ModeCassette(os.environ.get(ENV_CASSETTE, ModeCassette.LIVE.value))


# --------------------------------------------------------------------------- #
# Collecteurs de littérature
# --------------------------------------------------------------------------- #

TIMEOUT_HTTP_S = 30.0  # collecteurs PubMed, PMC, arXiv

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

RACINE_DEPOT = Path(__file__).resolve().parent.parent
DOSSIER_RUNS = RACINE_DEPOT / "runs"
DOSSIER_CATALOGUE = RACINE_DEPOT / "catalogue"
DOSSIER_CASSETTES_DEFAUT = RACINE_DEPOT / "tests" / "fixtures" / "cassettes"


def dossier_cassettes() -> Path:
    valeur = os.environ.get(ENV_DOSSIER_CASSETTES)
    return Path(valeur) if valeur else DOSSIER_CASSETTES_DEFAUT


# --------------------------------------------------------------------------- #
# Sandbox D1
# --------------------------------------------------------------------------- #


class BackendSandbox(StrEnum):
    DOCKER = "docker"
    BWRAP = "bwrap"
    UNSAFE_LOCAL = "unsafe_local"  # tests unitaires seulement, opt-in explicite


ENV_SANDBOX = "METIS_SANDBOX"


def backend_sandbox() -> BackendSandbox:
    return BackendSandbox(os.environ.get(ENV_SANDBOX, BackendSandbox.DOCKER.value))


# --------------------------------------------------------------------------- #
# Nettoyage des sources (versionné)
# --------------------------------------------------------------------------- #
# Incrémenté à toute modification de metis/literature/textclean.py ; enregistré
# dans chaque Document, car les offsets des claims visent le texte nettoyé.

TEXTCLEAN_VERSION = "2"  # v2 : résumé arXiv lu depuis l'entrée Atom brute


# --------------------------------------------------------------------------- #
# Lexiques du pré-filtre (versionnés)
# --------------------------------------------------------------------------- #
# Chaque expression commence et finit par \b ; prefilter.py les compile avec
# re.IGNORECASE et les applique au texte normalisé par verify.py (casse, NFKC,
# tirets et espaces unifiés). Toute modification incrémente LEXIQUES_VERSION
# et doit être suivie d'une nouvelle mesure des faux rejets sur le gold set.
# v2 : « adults » et « elderly » retirés des marqueurs humains (« adult mice »,
# « elderly rats ») ; un marqueur humain suivi d'un trait d'union ou de
# « derived » n'est pas compté (« patient-derived xenografts »), voir prefilter.py.
# v3 : numéro d'enregistrement collé au texte qui précède par la mise en forme PubMed
# (« Trial registration numberNCT02467153 », sans limite de mot avant NCT).

LEXIQUES_VERSION = "3"

LEXIQUE_ESPECES_NON_HUMAINES: tuple[str, ...] = (
    r"\bmice\b",
    r"\bmouse\b",
    r"\bmurine\b",
    r"\brats?\b",
    r"\brodents?\b",
    r"\bmarmosets?\b",
    r"\bmacaques?\b",
    r"\bmonkeys?\b",
    r"\brhesus\b",
    r"\b(non-human )?primates?\b",
    r"\bzebrafish\b",
    r"\bdrosophila\b",
    r"\bc\. elegans\b",
    r"\bcaenorhabditis\b",
    r"\byeast\b",
    r"\bhamsters?\b",
    r"\brabbits?\b",
    r"\bpigs?\b",
    r"\bporcine\b",
    r"\bdogs?\b",
)

LEXIQUE_SYSTEMES_NON_IN_VIVO: tuple[str, ...] = (
    r"\bin vitro\b",
    r"\bex vivo\b",
    r"\bin silico\b",
    r"\bcell cultures?\b",
    r"\bcultured cells\b",
    r"\bcell lines?\b",
    r"\borganoids?\b",
)

LEXIQUE_MARQUEURS_HUMAINS: tuple[str, ...] = (
    r"\bpatients?\b",
    r"\b(out|in)patients?\b",
    r"\bparticipants?\b",
    r"\bhuman subjects?\b",
    r"\bvolunteers?\b",
    r"\bcohorts?\b",
    r"\bindividuals\b",
    r"\bpeople\b",
    r"\bwomen\b",
    r"\bmen\b",
    r"\bchildren\b",
    r"\binfants?\b",
    r"\badolescents?\b",
)

# Garde mécanique de l'auditeur : une citation drapeau is_future_or_protocol sans
# aucun résultat chiffré ne peut pas être ESTABLISHED (rétrogradée CLAIMED_UNPROVEN).
# Ne comptent comme résultat chiffré que : valeur p, IC / CI (y compris « confidence
# interval »), HR, OR, RR, MD, et ± (ou sa notation ASCII « +/- »). Un nombre suivi
# d'une unité ne compte pas : doses et effectifs (« 2 mg », « 140 patients ») figurent
# dans les protocoles. Sensible à la casse, sauf pour p et « confidence interval ».
MOTIF_RESULTAT_CHIFFRE = (
    r"\b[pP]\s*[<=>≤≥]"
    r"|\b(?:CI|IC)\b"
    r"|\b[Cc]onfidence [Ii]nterval\b"
    r"|\b(?:HR|OR|RR|MD)\b(?=\s*[=:,(\[]|\s+-?\d|\s+of\b)"
    r"|±|\+/-"
)
# Retirés avant la recherche : un calcul de puissance n'est pas un résultat.
MOTIF_PUISSANCE_EXCLUE = r"\d+(?:[.,]\d+)?\s*%\s*power\b|\bpower of\b[^.;]*"

# Au-dessous de ce nombre de caractères (un abstract), l'auditeur reçoit le document
# entier, citation balisée ; au-dessus (texte intégral), les fenêtres et le résumé.
SEUIL_DOCUMENT_ENTIER = 6_000

LEXIQUE_FUTUR_PROTOCOLE: tuple[str, ...] = (
    r"\bwill be (analy[sz]ed|evaluated|assessed|measured|randomi[sz]ed|enrolled|recruited)\b",
    r"\bwill (evaluate|assess|investigate|determine|examine)\b",
    r"\bis being (evaluated|tested)\b",
    r"\bare being (recruited|enrolled)\b",
    r"\bis (currently )?recruiting\b",
    r"\bongoing (trial|study)\b",
    r"\bprotocol\b",
    r"\bNCT\d{8}\b",
    r"\bregistration number\s*:?\s*NCT\d{8}\b",
)


# --------------------------------------------------------------------------- #
# Collecteurs
# --------------------------------------------------------------------------- #

EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print"
ENV_NCBI_EMAIL = "METIS_NCBI_EMAIL"  # recommandé par NCBI, facultatif
PAUSE_NCBI_S = 0.34  # 3 requêtes/s sans clé NCBI
PAUSE_ARXIV_S = 3.0  # recommandation arXiv
MAX_RESULTATS_REQUETE_INITIALE = 10
MAX_RESULTATS_REQUETE_ADVERSE = 5
MAX_OCTETS_SOURCE_LATEX = 5_000_000


def ncbi_email() -> str | None:
    return os.environ.get(ENV_NCBI_EMAIL) or None


# --------------------------------------------------------------------------- #
# Pipeline D0
# --------------------------------------------------------------------------- #

MAX_CARACTERES_PAR_APPEL = 40_000  # au-delà, le document est découpé en segments
CHEVAUCHEMENT_SEGMENTS = 1_000
MAX_CLAIMS_PAR_APPEL = 8
MAX_EQUATIONS_PAR_APPEL = 3
MAX_REQUETES_ADVERSES = 3
MAX_PREUVES_PAR_DOCUMENT = 3
