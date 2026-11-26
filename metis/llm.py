"""Client OpenRouter à sorties structurées, RunLedger et cassettes.

Aucun état global : chaque appel reçoit explicitement le ClientLLM et le RunLedger
du run. Le modèle est déduit du rôle via config.py ; aucun nom de modèle ici.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from metis import config
from metis.config import ModeCassette, Tarif
from metis.schemas import Cassette, CoutsRun, EntreeLedger, Message, RoleLLM
from metis.storage import ecrire_json_atomique, json_canonique, lire_json

T = TypeVar("T", bound=BaseModel)


class UnknownModelPricingError(Exception):
    """Modèle absent de config.TABLE_PRIX ; levée avant tout appel."""


class CassetteMissError(Exception):
    """Mode replay et aucune cassette pour cette clé ; jamais de repli réseau."""


class UsageManquantError(Exception):
    """Réponse sans champ usage exploitable ; les tokens ne sont jamais estimés."""


class SortieLLMInvalideError(Exception):
    """Sortie non conforme au schéma après l'unique retry de réparation."""

    motif = "juge_sortie_invalide"

    def __init__(self, role: RoleLLM, erreurs: str) -> None:
        super().__init__(f"{role} : sortie invalide après réparation : {erreurs}")
        self.role = role
        self.erreurs = erreurs


class RefusFournisseurError(SortieLLMInvalideError):
    """Le fournisseur a refusé de répondre (finish_reason content_filter / refusal).

    Ce n'est pas une sortie hors schéma : aucun retry de réparation n'est tenté (il
    serait facturé pour rien) et le motif fail-closed est distinct.
    """

    motif = "refus_fournisseur"

    def __init__(self, role: RoleLLM, detail: str) -> None:
        Exception.__init__(self, f"{role} : refus du fournisseur ({detail})")
        self.role = role
        self.erreurs = detail


class BudgetDepasseError(Exception):
    """L'appel pourrait faire dépasser METIS_BUDGET_USD au run ; il n'est pas envoyé."""


class CleApiManquanteError(Exception):
    """Appel réseau demandé sans clé OpenRouter dans l'environnement."""


# --------------------------------------------------------------------------- #
# Prix et ledger
# --------------------------------------------------------------------------- #


def tarif(modele: str) -> Tarif:
    if modele in config.TABLE_PRIX:
        return config.TABLE_PRIX[modele]
    if config.est_modele_local(modele):
        return Tarif(0.0, 0.0)
    raise UnknownModelPricingError(f"aucun tarif pour le modèle {modele!r}") from None


def _est_hote_local(url: str) -> bool:
    hote = urlparse(url).hostname
    return hote in config.HOTES_LOCAUX if hote is not None else False


def cout_usd(t: Tarif, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * t.entree_usd_par_mtok + completion_tokens * t.sortie_usd_par_mtok) / 1e6


def cout_maximal_usd(t: Tarif, messages: Sequence[Message], schema: type[BaseModel], max_tokens: int) -> float:
    """Borne supérieure du coût d'un appel (pas une estimation de l'usage réel).

    Un token couvre au moins un octet : les octets UTF-8 des messages et du schéma,
    plus une marge par message, majorent les tokens d'entrée ; la sortie est bornée
    par max_tokens (plafond de sortie du rôle).
    """
    octets = sum(len(m.content.encode("utf-8")) + 64 for m in messages)
    octets += len(json_canonique(schema.model_json_schema())) + 512
    return cout_usd(t, octets, max_tokens)


class RunLedger:
    """Journal des coûts d'un run ; réécrit atomiquement couts.json après chaque appel.

    Archive aussi chaque réponse brute (tous modes) dans runs/<run_id>/reponses_llm/.
    """

    def __init__(self, run_id: str, dossier_run: Path, budget_usd: float | None = None) -> None:
        self.run_id = run_id
        self.chemin = dossier_run / "couts.json"
        self.dossier_reponses = dossier_run / "reponses_llm"
        self.budget_usd = config.budget_usd() if budget_usd is None else budget_usd
        self._entrees: list[EntreeLedger] = []
        self._reserve_usd = 0.0
        self._nb_reponses = 0
        self._verrou = threading.Lock()

    def depense_reelle_usd(self) -> float:
        """Coût des appels réellement envoyés (hors rejeux de cassettes)."""
        with self._verrou:
            return sum(e.cost_usd for e in self._entrees if not e.cassette)

    @contextmanager
    def reserver(self, cout_maximal_usd: float) -> Iterator[None]:
        """Réserve le coût maximal d'un appel réel ; BudgetDepasseError s'il ne tient pas."""
        with self._verrou:
            depense = sum(e.cost_usd for e in self._entrees if not e.cassette)
            if depense + self._reserve_usd + cout_maximal_usd > self.budget_usd:
                raise BudgetDepasseError(
                    f"run {self.run_id} : {depense:.4f} USD dépensés + {cout_maximal_usd:.4f} USD possibles "
                    f"> budget {self.budget_usd:.2f} USD"
                )
            self._reserve_usd += cout_maximal_usd
        try:
            yield
        finally:
            with self._verrou:
                self._reserve_usd -= cout_maximal_usd

    @property
    def entrees(self) -> tuple[EntreeLedger, ...]:
        with self._verrou:
            return tuple(self._entrees)

    def instantane(self) -> CoutsRun:
        entrees = self.entrees
        return CoutsRun(
            run_id=self.run_id,
            entrees=entrees,
            cout_total_usd=sum(e.cost_usd for e in entrees),
        )

    def archiver_reponse(self, role: RoleLLM, cassette: Cassette) -> str:
        """Écrit la réponse brute avant toute analyse ; renvoie son chemin relatif au run."""
        with self._verrou:
            self._nb_reponses += 1
            nom = f"{self._nb_reponses:04d}_{role.value}_{cassette.cle[:12]}.json"
        ecrire_json_atomique(self.dossier_reponses / nom, cassette)
        return f"{self.dossier_reponses.name}/{nom}"

    def enregistrer(self, entree: EntreeLedger) -> None:
        with self._verrou:
            self._entrees.append(entree)
            ecrire_json_atomique(
                self.chemin,
                CoutsRun(
                    run_id=self.run_id,
                    entrees=tuple(self._entrees),
                    cout_total_usd=sum(e.cost_usd for e in self._entrees),
                ),
            )


# --------------------------------------------------------------------------- #
# Cassettes
# --------------------------------------------------------------------------- #


def cle_cassette(
    modele: str,
    messages: Sequence[Message],
    schema: type[BaseModel],
    fournisseurs: tuple[str, ...] | None = None,
) -> str:
    charge: dict[str, object] = {
        "modele": modele,
        "messages": [m.model_dump(mode="json") for m in messages],
        "schema": schema.model_json_schema(),
    }
    if fournisseurs:  # absent sinon : les clés sans épinglage restent inchangées
        charge["fournisseurs"] = list(fournisseurs)
    return hashlib.sha256(json_canonique(charge)).hexdigest()


def chemin_cassette(dossier: Path, cle: str) -> Path:
    return dossier / f"{cle}.json"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class ClientLLM:
    def __init__(
        self,
        mode: ModeCassette,
        dossier_cassettes: Path,
        http: httpx.Client | None = None,
    ) -> None:
        self.mode = mode
        self.dossier_cassettes = dossier_cassettes
        self._http = http

    @classmethod
    def depuis_environnement(cls) -> ClientLLM:
        return cls(config.mode_cassette(), config.dossier_cassettes())

    def completer(
        self,
        ledger: RunLedger,
        role: RoleLLM,
        messages: Sequence[Message],
        schema: type[BaseModel],
    ) -> str:
        """Un appel (ou un rejeu) ; journalise le coût ; renvoie le contenu brut."""
        modele = config.modele_pour_role(role)
        t = tarif(modele)  # avant toute E/S
        fournisseurs = config.fournisseurs_pour_role(role)
        cle = cle_cassette(modele, messages, schema, fournisseurs)

        chemin = chemin_cassette(self.dossier_cassettes, cle)
        if self.mode is ModeCassette.REPLAY or (self.mode is ModeCassette.RECORD_MISSING and chemin.exists()):
            if not chemin.exists():
                raise CassetteMissError(f"cassette absente pour {role} / {modele} : {cle}")
            cassette = Cassette.model_validate(lire_json(chemin))
            return self._journaliser(ledger, role, t, cassette, rejouee=True)
        # La réservation couvre l'appel et son enregistrement : aucun appel concurrent ne
        # peut se glisser entre les deux et faire dépasser le budget.
        max_tokens = config.max_tokens_pour_role(role)
        with ledger.reserver(cout_maximal_usd(t, messages, schema, max_tokens)):
            debut = time.perf_counter()
            reponse = self._poster(modele, messages, schema, fournisseurs, max_tokens)
            cassette = Cassette(
                cle=cle,
                modele=modele,
                date_enregistrement=datetime.now(UTC),
                nom_schema=schema.__name__,
                messages=tuple(messages),
                latency_ms=(time.perf_counter() - debut) * 1000.0,
                reponse_brute=reponse,
                fournisseur=_fournisseur(reponse),
            )
            if self.mode in (ModeCassette.RECORD, ModeCassette.RECORD_MISSING):
                ecrire_json_atomique(chemin, cassette)
            return self._journaliser(ledger, role, t, cassette, rejouee=False)

    def _journaliser(self, ledger: RunLedger, role: RoleLLM, t: Tarif, cassette: Cassette, rejouee: bool) -> str:
        fichier = ledger.archiver_reponse(role, cassette)  # avant toute analyse
        prompt_tokens, completion_tokens = _usage(cassette.reponse_brute)
        refus = _refus(cassette.reponse_brute)
        ledger.enregistrer(
            EntreeLedger(
                role=role,
                model=cassette.modele,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cout_usd(t, prompt_tokens, completion_tokens),
                latency_ms=cassette.latency_ms,
                cassette=rejouee,
                horodatage=datetime.now(UTC),
                fichier_reponse=fichier,
                refus_fournisseur=refus is not None,
                fournisseur=cassette.fournisseur or _fournisseur(cassette.reponse_brute),
            )
        )
        if refus is not None:
            raise RefusFournisseurError(role, refus)
        return _contenu(cassette.reponse_brute)

    def _poster(
        self,
        modele: str,
        messages: Sequence[Message],
        schema: type[BaseModel],
        fournisseurs: tuple[str, ...] | None,
        max_tokens: int,
    ) -> JsonValue:
        url = config.llm_base_url()
        est_local = config.est_modele_local(modele) and _est_hote_local(url)
        cle_api = os.environ.get(config.ENV_OPENROUTER_CLE)
        if not cle_api and not est_local:
            raise CleApiManquanteError(f"{config.ENV_OPENROUTER_CLE} non défini")
        if self._http is None:
            self._http = httpx.Client(timeout=config.TIMEOUT_LLM_S)
        corps: dict[str, object] = {
            "model": config.nom_modele_distant(modele) if est_local else modele,
            "messages": [m.model_dump(mode="json") for m in messages],
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": schema.model_json_schema()},
            },
        }
        if fournisseurs and not est_local:
            corps["provider"] = {"order": list(fournisseurs), "allow_fallbacks": False}
        entetes = {"Authorization": f"Bearer {cle_api}"} if cle_api else {}
        reponse = self._http.post(url, json=corps, headers=entetes)
        reponse.raise_for_status()
        donnees: JsonValue = reponse.json()
        return donnees


def _usage(reponse: JsonValue) -> tuple[int, int]:
    usage = reponse.get("usage") if isinstance(reponse, dict) else None
    if not isinstance(usage, dict):
        raise UsageManquantError("champ usage absent de la réponse")
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int) or prompt < 0 or completion < 0:
        raise UsageManquantError(f"usage invalide : {usage!r}")
    return prompt, completion


def _fournisseur(reponse: JsonValue) -> str | None:
    fournisseur = reponse.get("provider") if isinstance(reponse, dict) else None
    return fournisseur if isinstance(fournisseur, str) else None


def _refus(reponse: JsonValue) -> str | None:
    """Description du refus si le fournisseur a filtré la réponse (contenu absent ou tronqué)."""
    if not isinstance(reponse, dict):
        return None
    choix = reponse.get("choices")
    if not isinstance(choix, list) or not choix or not isinstance(choix[0], dict):
        return None
    fin, fin_native = choix[0].get("finish_reason"), choix[0].get("native_finish_reason")
    if fin == "content_filter" or fin_native == "refusal":
        return f"finish_reason={fin}, native_finish_reason={fin_native}, fournisseur={reponse.get('provider')}"
    return None


def _contenu(reponse: JsonValue) -> str:
    """Texte de la première réponse ; chaîne vide si absent (refus, format inattendu)."""
    if not isinstance(reponse, dict):
        return ""
    choix = reponse.get("choices")
    if not isinstance(choix, list) or not choix or not isinstance(choix[0], dict):
        return ""
    message = choix[0].get("message")
    contenu = message.get("content") if isinstance(message, dict) else None
    return contenu if isinstance(contenu, str) else ""


# --------------------------------------------------------------------------- #
# Sorties structurées avec un retry de réparation
# --------------------------------------------------------------------------- #


def appeler_structure(
    client: ClientLLM,
    ledger: RunLedger,
    role: RoleLLM,
    messages: Sequence[Message],
    schema: type[T],
    controle: Callable[[T], str | None] | None = None,
) -> T:
    """Appelle le modèle du rôle et valide sa sortie contre schema.

    controle : vérification supplémentaire du harnais (ex. une évaluation par fenêtre),
    traitée comme une erreur de validation. Une seule réparation : l'erreur est
    renvoyée au modèle. Au second échec, SortieLLMInvalideError ; l'appelant applique
    la dégradation fail-closed. Un refus du fournisseur lève RefusFournisseurError
    sans réparation.
    """
    contenu = client.completer(ledger, role, messages, schema)
    premier = _valider(contenu, schema, controle)
    if isinstance(premier, BaseModel):
        return premier
    reparation = (
        *messages,
        Message(role="assistant", content=contenu),
        Message(
            role="user",
            content=(
                "Ta réponse ne respecte pas le schéma JSON demandé. Erreurs de validation :\n"
                f"{premier}\nRenvoie uniquement l'objet JSON corrigé, sans autre texte."
            ),
        ),
    )
    contenu = client.completer(ledger, role, reparation, schema)
    resultat = _valider(contenu, schema, controle)
    if isinstance(resultat, BaseModel):
        return resultat
    raise SortieLLMInvalideError(role, resultat)


def _valider(contenu: str, schema: type[T], controle: Callable[[T], str | None] | None) -> T | str:
    try:
        sortie = schema.model_validate_json(contenu)
    except ValidationError as erreur:
        return str(erreur)
    probleme = controle(sortie) if controle is not None else None
    return sortie if probleme is None else probleme
