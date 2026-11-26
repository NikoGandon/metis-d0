"""Étape 1 : écriture atomique, RunLedger, prix, machines à états, cassettes."""

from __future__ import annotations

import json
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from metis import config
from metis.config import ModeCassette
from metis.llm import (
    BudgetDepasseError,
    CassetteMissError,
    ClientLLM,
    RefusFournisseurError,
    RunLedger,
    SortieLLMInvalideError,
    UnknownModelPricingError,
    UsageManquantError,
    appeler_structure,
    chemin_cassette,
    cle_cassette,
    cout_maximal_usd,
    tarif,
)
from metis.schemas import (
    Cassette,
    CommandeD1,
    Claim,
    CoutsRun,
    Drapeaux,
    EntreeLedger,
    InvalidTransitionError,
    Message,
    MotifRefus,
    Occurrence,
    Offsets,
    Porte,
    PreuveAdverse,
    RoleLLM,
    StatutClaim,
    StatutPreuve,
    Verdict,
    VerdictRoutage,
)
from metis.storage import ecrire_json_atomique, json_canonique, lire_json

# --------------------------------------------------------------------------- #
# storage.py
# --------------------------------------------------------------------------- #


def test_ecriture_atomique_contenu_canonique(tmp_path: Path) -> None:
    chemin = tmp_path / "sous" / "dossier" / "f.json"
    ecrire_json_atomique(chemin, {"b": 1, "a": "é"})
    brut = chemin.read_bytes()
    assert brut == '{\n  "a": "é",\n  "b": 1\n}\n'.encode()
    assert lire_json(chemin) == {"a": "é", "b": 1}
    assert list(chemin.parent.iterdir()) == [chemin]


def test_crash_avant_replace_laisse_ancien_fichier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chemin = tmp_path / "f.json"
    ecrire_json_atomique(chemin, {"version": 1})
    ancien = chemin.read_bytes()

    def crash(*_a: Any, **_k: Any) -> None:
        raise OSError("crash simulé avant os.replace")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="crash simulé"):
        ecrire_json_atomique(chemin, {"version": 2})
    assert chemin.read_bytes() == ancien
    assert list(tmp_path.iterdir()) == [chemin]  # temporaire nettoyé


def test_crash_pendant_fsync_laisse_ancien_fichier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chemin = tmp_path / "f.json"
    ecrire_json_atomique(chemin, {"version": 1})
    ancien = chemin.read_bytes()

    def crash(_fd: int) -> None:
        raise OSError("disque plein")

    monkeypatch.setattr(os, "fsync", crash)
    with pytest.raises(OSError):
        ecrire_json_atomique(chemin, {"version": 2})
    assert chemin.read_bytes() == ancien
    assert list(tmp_path.iterdir()) == [chemin]


def test_objet_non_serialisable_ne_touche_pas_le_disque(tmp_path: Path) -> None:
    chemin = tmp_path / "f.json"
    ecrire_json_atomique(chemin, {"ok": True})
    with pytest.raises(ValueError):
        ecrire_json_atomique(chemin, {"x": math.nan})
    with pytest.raises(TypeError):
        ecrire_json_atomique(chemin, {"x": object()})
    assert lire_json(chemin) == {"ok": True}
    assert list(tmp_path.iterdir()) == [chemin]


def test_json_canonique_independant_de_l_ordre() -> None:
    assert json_canonique({"b": [1, 2], "a": {"d": 1, "c": 2}}) == json_canonique(
        {"a": {"c": 2, "d": 1}, "b": [1, 2]}
    )
    assert json_canonique({"a": 1}) == b'{"a":1}'


# --------------------------------------------------------------------------- #
# RunLedger et prix
# --------------------------------------------------------------------------- #


def _entree(role: RoleLLM, n: int) -> EntreeLedger:
    return EntreeLedger(
        role=role,
        model=config.MODELE_JUGE_DEFAUT,
        prompt_tokens=n,
        completion_tokens=n,
        cost_usd=n / 1e6,
        latency_ms=1.0,
        cassette=False,
        horodatage=datetime.now(UTC),
    )


def test_ledgers_concurrents_ne_se_melangent_pas(tmp_path: Path) -> None:
    ledgers = {
        RoleLLM.EXTRACTEUR: RunLedger("run-a", tmp_path / "run-a"),
        RoleLLM.FORGERON: RunLedger("run-b", tmp_path / "run-b"),
    }
    n_threads, n_appels = 4, 50

    def travailler(role: RoleLLM) -> None:
        for i in range(n_appels):
            ledgers[role].enregistrer(_entree(role, i))

    threads = [threading.Thread(target=travailler, args=(role,)) for role in ledgers for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for role, ledger in ledgers.items():
        persiste = CoutsRun.model_validate(lire_json(ledger.chemin))
        assert persiste.run_id == ledger.run_id
        assert len(persiste.entrees) == n_threads * n_appels
        assert {e.role for e in persiste.entrees} == {role}
        assert persiste == ledger.instantane()
        attendu = n_threads * sum(range(n_appels)) / 1e6
        assert persiste.cout_total_usd == pytest.approx(attendu)


class _Sortie(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    valeur: int


def _messages() -> tuple[Message, ...]:
    return (Message(role="system", content="Réponds en JSON."), Message(role="user", content="Donne 3."))


class _TransportEspion:
    """Transport httpx factice : enregistre les requêtes, rejoue une liste de réponses."""

    def __init__(self, reponses: list[dict[str, Any]]) -> None:
        self.reponses = list(reponses)
        self.requetes: list[dict[str, Any]] = []

    def __call__(self, requete: httpx.Request) -> httpx.Response:
        self.requetes.append(json.loads(requete.content))
        return httpx.Response(200, json=self.reponses.pop(0))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def _reponse(contenu: str, prompt: int = 1000, completion: int = 200) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": contenu}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


@pytest.mark.parametrize("mode", list(ModeCassette))
def test_modele_sans_tarif_leve_avant_tout_appel(
    mode: ModeCassette, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.ENV_MODELE_JUGE, "fournisseur/modele-inconnu")
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    espion = _TransportEspion([_reponse('{"valeur": 3}')])
    client = ClientLLM(mode, tmp_path / "cassettes", http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")

    with pytest.raises(UnknownModelPricingError):
        appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert espion.requetes == []
    assert ledger.entrees == ()
    assert not ledger.chemin.exists()
    assert not (tmp_path / "cassettes").exists()


def test_modele_ouvrier_pour_le_forgeron(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_MODELE_JUGE, "juge/x")
    monkeypatch.setenv(config.ENV_MODELE_OUVRIER, "ouvrier/y")
    assert config.modele_pour_role(RoleLLM.FORGERON) == "ouvrier/y"
    for role in set(RoleLLM) - {RoleLLM.FORGERON}:
        assert config.modele_pour_role(role) == "juge/x"


def test_modele_par_role_avec_juge_par_defaut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_MODELE_JUGE, "juge/x")
    for variable in config.ENV_MODELE_PAR_ROLE.values():
        monkeypatch.delenv(variable, raising=False)
    assert {config.modele_pour_role(r) for r in config.ENV_MODELE_PAR_ROLE} == {"juge/x"}
    monkeypatch.setenv("METIS_MODEL_AUDITEUR", "auditeur/y")
    assert config.modele_pour_role(RoleLLM.AUDITEUR) == "auditeur/y"
    assert config.modele_pour_role(RoleLLM.ROUTEUR) == "juge/x"


def test_defaut_opus_pour_tous_les_juges(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (config.ENV_MODELE_JUGE, *config.ENV_MODELE_PAR_ROLE.values()):
        monkeypatch.delenv(variable, raising=False)
    assert {config.modele_pour_role(r) for r in config.ENV_MODELE_PAR_ROLE} == {config.MODELE_JUGE_DEFAUT}


def test_epinglage_fournisseur_et_fournisseur_effectif(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    monkeypatch.setenv("METIS_PROVIDER_AUDITEUR", "anthropic, amazon-bedrock")
    reponse = {**_reponse('{"valeur": 3}'), "provider": "Anthropic"}
    espion = _TransportEspion([reponse])
    client = ClientLLM(ModeCassette.RECORD, tmp_path / "cassettes", http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")
    appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    (requete,) = espion.requetes
    assert requete["provider"] == {"order": ["anthropic", "amazon-bedrock"], "allow_fallbacks": False}
    (entree,) = ledger.entrees
    assert entree.fournisseur == "Anthropic"
    (fichier,) = (tmp_path / "cassettes").iterdir()
    assert Cassette.model_validate(lire_json(fichier)).fournisseur == "Anthropic"
    # l'épinglage fait partie de la clé ; sans épinglage, la clé est celle d'avant
    assert fichier.stem == cle_cassette(config.modele_juge(), _messages(), _Sortie, ("anthropic", "amazon-bedrock"))
    assert cle_cassette(config.modele_juge(), _messages(), _Sortie) == cle_cassette(
        config.modele_juge(), _messages(), _Sortie, None
    )


def test_sans_epinglage_aucun_champ_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    monkeypatch.delenv("METIS_PROVIDER_AUDITEUR", raising=False)
    espion = _TransportEspion([_reponse('{"valeur": 3}')])
    appeler_structure(ClientLLM(ModeCassette.LIVE, tmp_path, http=espion.client()),
                      RunLedger("r", tmp_path / "r"), RoleLLM.AUDITEUR, _messages(), _Sortie)  # fmt: skip
    assert "provider" not in espion.requetes[0]


def test_aucun_nom_de_modele_hors_config() -> None:
    racine = Path(config.__file__).parent
    noms = list(config.TABLE_PRIX) + [config.MODELE_JUGE_DEFAUT, config.MODELE_OUVRIER_DEFAUT]
    for fichier in racine.rglob("*.py"):
        if fichier.name == "config.py":
            continue
        texte = fichier.read_text(encoding="utf-8")
        for nom in noms:
            assert nom not in texte, f"{nom} apparaît dans {fichier}"


# --------------------------------------------------------------------------- #
# Cassettes et appels structurés
# --------------------------------------------------------------------------- #


def _ecrire_cassette(dossier: Path, role: RoleLLM, messages: tuple[Message, ...], reponse: dict[str, Any]) -> None:
    modele = config.modele_pour_role(role)
    cle = cle_cassette(modele, messages, _Sortie)
    ecrire_json_atomique(
        chemin_cassette(dossier, cle),
        Cassette(
            cle=cle,
            modele=modele,
            date_enregistrement=datetime(2026, 11, 3, tzinfo=UTC),
            nom_schema="_Sortie",
            messages=messages,
            latency_ms=1234.0,
            reponse_brute=reponse,
        ),
    )


def test_cassette_absente_leve_cassette_miss(tmp_path: Path) -> None:
    client = ClientLLM(ModeCassette.REPLAY, tmp_path)
    ledger = RunLedger("run-x", tmp_path / "run-x")
    with pytest.raises(CassetteMissError):
        appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert ledger.entrees == ()


def test_mode_replay_par_defaut_sous_pytest() -> None:
    assert config.mode_cassette() is ModeCassette.REPLAY
    assert ClientLLM.depuis_environnement().mode is ModeCassette.REPLAY


def test_reseau_bloque_sous_pytest() -> None:
    with pytest.raises(RuntimeError, match="réseau"):
        httpx.get("https://example.org")


def test_rejeu_cassette_et_ledger(tmp_path: Path) -> None:
    _ecrire_cassette(tmp_path, RoleLLM.AUDITEUR, _messages(), _reponse('{"valeur": 3}'))
    client = ClientLLM(ModeCassette.REPLAY, tmp_path)
    ledger = RunLedger("run-x", tmp_path / "run-x")

    assert appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie) == _Sortie(valeur=3)
    (entree,) = ledger.entrees
    assert entree.cassette is True
    assert entree.role is RoleLLM.AUDITEUR
    assert entree.model == config.modele_juge()
    assert (entree.prompt_tokens, entree.completion_tokens) == (1000, 200)
    t = config.TABLE_PRIX[config.modele_juge()]
    assert entree.cost_usd == pytest.approx((1000 * t.entree_usd_par_mtok + 200 * t.sortie_usd_par_mtok) / 1e6)
    assert entree.latency_ms == 1234.0
    assert CoutsRun.model_validate(lire_json(ledger.chemin)).entrees == (entree,)


def test_cle_cassette_depend_du_modele_du_prompt_et_du_schema() -> None:
    class _Autre(BaseModel):
        valeur: str

    base = cle_cassette("m/a", _messages(), _Sortie)
    assert base == cle_cassette("m/a", _messages(), _Sortie)
    assert base != cle_cassette("m/b", _messages(), _Sortie)
    assert base != cle_cassette("m/a", _messages()[:1], _Sortie)
    assert base != cle_cassette("m/a", _messages(), _Autre)


def test_record_puis_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    espion = _TransportEspion([_reponse('{"valeur": 7}')])
    enregistreur = ClientLLM(ModeCassette.RECORD, tmp_path, http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")
    assert appeler_structure(enregistreur, ledger, RoleLLM.ROUTEUR, _messages(), _Sortie).valeur == 7

    (requete,) = espion.requetes
    assert requete["model"] == config.modele_juge()
    assert requete["response_format"]["json_schema"]["schema"] == _Sortie.model_json_schema()
    (fichier,) = [p for p in tmp_path.iterdir() if p.suffix == ".json"]
    cassette = Cassette.model_validate(lire_json(fichier))
    assert cassette.messages == _messages()

    lecteur = ClientLLM(ModeCassette.REPLAY, tmp_path)
    assert appeler_structure(lecteur, ledger, RoleLLM.ROUTEUR, _messages(), _Sortie).valeur == 7
    assert [e.cassette for e in ledger.entrees] == [False, True]


def test_un_seul_retry_de_reparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    espion = _TransportEspion([_reponse("pas du json"), _reponse('{"valeur": 3}')])
    client = ClientLLM(ModeCassette.LIVE, tmp_path, http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")

    assert appeler_structure(client, ledger, RoleLLM.EXTRACTEUR, _messages(), _Sortie).valeur == 3
    assert len(espion.requetes) == 2
    reparation = espion.requetes[1]["messages"]
    assert reparation[-2] == {"role": "assistant", "content": "pas du json"}
    assert "Erreurs de validation" in reparation[-1]["content"]
    assert len(ledger.entrees) == 2  # les deux appels sont facturés


def test_deux_sorties_invalides_levent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    espion = _TransportEspion([_reponse('{"valeur": "x"}'), _reponse('{"valeur": 1, "extra": 2}')])
    client = ClientLLM(ModeCassette.LIVE, tmp_path, http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")

    with pytest.raises(SortieLLMInvalideError) as info:
        appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert info.value.role is RoleLLM.AUDITEUR
    assert len(espion.requetes) == 2
    assert len(ledger.entrees) == 2


@pytest.mark.parametrize("mode", [ModeCassette.LIVE, ModeCassette.RECORD, ModeCassette.REPLAY])
def test_reponses_brutes_archivees_dans_le_run(
    mode: ModeCassette, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    brute = _reponse('{"valeur": 5}')
    _ecrire_cassette(tmp_path / "cassettes", RoleLLM.AUDITEUR, _messages(), brute)
    espion = _TransportEspion([brute])
    client = ClientLLM(mode, tmp_path / "cassettes", http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")

    appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    (entree,) = ledger.entrees
    assert entree.fichier_reponse is not None
    archive = Cassette.model_validate(lire_json(tmp_path / "run-x" / entree.fichier_reponse))
    assert archive.reponse_brute == brute
    assert archive.messages == _messages()
    assert archive.modele == entree.model


def test_reponse_sans_usage_archivee_quand_meme(tmp_path: Path) -> None:
    sans_usage = {"choices": [{"message": {"content": '{"valeur": 1}'}}]}
    _ecrire_cassette(tmp_path, RoleLLM.AUDITEUR, _messages(), sans_usage)
    ledger = RunLedger("run-x", tmp_path / "run-x")
    with pytest.raises(UsageManquantError):
        appeler_structure(ClientLLM(ModeCassette.REPLAY, tmp_path), ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    (fichier,) = (tmp_path / "run-x" / "reponses_llm").iterdir()
    assert Cassette.model_validate(lire_json(fichier)).reponse_brute == sans_usage


def test_record_missing_rejoue_l_existant_et_enregistre_le_manquant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    _ecrire_cassette(tmp_path, RoleLLM.AUDITEUR, _messages(), _reponse('{"valeur": 1}'))
    autres = (*_messages()[:1], Message(role="user", content="Donne 2."))
    espion = _TransportEspion([_reponse('{"valeur": 2}')])
    client = ClientLLM(ModeCassette.RECORD_MISSING, tmp_path, http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")

    assert appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie).valeur == 1
    assert espion.requetes == []  # existante : rejouée
    assert appeler_structure(client, ledger, RoleLLM.AUDITEUR, autres, _Sortie).valeur == 2
    assert len(espion.requetes) == 1  # manquante : appel réel puis cassette
    assert chemin_cassette(tmp_path, cle_cassette(config.modele_juge(), autres, _Sortie)).exists()
    assert [e.cassette for e in ledger.entrees] == [True, False]


def _refus_fournisseur(contenu: str | None) -> dict[str, Any]:
    return {
        "choices": [{"finish_reason": "content_filter", "native_finish_reason": "refusal",
                     "message": {"role": "assistant", "content": contenu}}],
        "usage": {"prompt_tokens": 2453, "completion_tokens": 0},
        "provider": "Claude Platform on AWS",
    }  # fmt: skip


@pytest.mark.parametrize("contenu", [None, '{"valeur": 3'])
def test_refus_fournisseur_sans_retry_de_reparation(
    contenu: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    espion = _TransportEspion([_refus_fournisseur(contenu), _reponse('{"valeur": 3}')])
    client = ClientLLM(ModeCassette.LIVE, tmp_path, http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")
    with pytest.raises(RefusFournisseurError) as info:
        appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert info.value.motif == "refus_fournisseur"
    assert "native_finish_reason=refusal" in info.value.erreurs
    assert len(espion.requetes) == 1  # pas de réparation facturée pour rien
    (entree,) = ledger.entrees
    assert entree.prompt_tokens == 2453  # le refus reste facturé et journalisé


# --------------------------------------------------------------------------- #
# Budget par run
# --------------------------------------------------------------------------- #


def test_budget_par_defaut_et_surcharge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(config.ENV_BUDGET, raising=False)
    assert RunLedger("r", tmp_path).budget_usd == 10.0
    monkeypatch.setenv(config.ENV_BUDGET, "2.5")
    assert RunLedger("r", tmp_path).budget_usd == 2.5
    for invalide in ("0", "-1", "inf", "nan", "dix"):
        monkeypatch.setenv(config.ENV_BUDGET, invalide)
        with pytest.raises(ValueError):
            RunLedger("r", tmp_path)


def test_appel_refuse_avant_envoi_si_budget_insuffisant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    espion = _TransportEspion([_reponse('{"valeur": 3}')])
    client = ClientLLM(ModeCassette.LIVE, tmp_path, http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x", budget_usd=0.01)
    with pytest.raises(BudgetDepasseError):
        appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert espion.requetes == [] and ledger.entrees == ()


def test_budget_cumule_sur_le_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    t = tarif(config.modele_juge())
    maximum = cout_maximal_usd(t, _messages(), _Sortie, config.max_tokens_pour_role(RoleLLM.AUDITEUR))
    espion = _TransportEspion([_reponse('{"valeur": 3}'), _reponse('{"valeur": 4}')])
    client = ClientLLM(ModeCassette.LIVE, tmp_path, http=espion.client())
    reel = cout_usd_reference(t, 1000, 200)  # usage renvoyé par _reponse()
    ledger = RunLedger("run-x", tmp_path / "run-x", budget_usd=maximum + reel / 2)
    appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)  # 0 + max ≤ budget
    assert ledger.depense_reelle_usd() > 0
    with pytest.raises(BudgetDepasseError):  # dépensé + max > budget
        appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert len(espion.requetes) == 1
    assert ledger.depense_reelle_usd() <= ledger.budget_usd


def test_borne_de_cout_majore_le_cout_reel() -> None:
    t = tarif(config.modele_juge())
    octets = sum(len(m.content.encode()) for m in _messages())
    plafond = config.max_tokens_pour_role(RoleLLM.AUDITEUR)
    assert cout_maximal_usd(t, _messages(), _Sortie, plafond) >= cout_usd_reference(t, octets, plafond)


def test_max_tokens_par_role_hors_cle_de_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert set(config.MAX_TOKENS_PAR_ROLE) == set(RoleLLM)
    assert config.max_tokens_pour_role(RoleLLM.AUDITEUR) == 2_000
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, "cle-de-test")
    espion = _TransportEspion([_reponse('{"valeur": 3}'), _reponse('{"valeur": 4}')])
    client = ClientLLM(ModeCassette.RECORD, tmp_path / "c", http=espion.client())
    ledger = RunLedger("run-x", tmp_path / "run-x")
    appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert espion.requetes[0]["max_tokens"] == 2_000
    (fichier,) = (tmp_path / "c").iterdir()
    assert fichier.stem == cle_cassette(config.modele_juge(), _messages(), _Sortie)  # max_tokens absent de la clé
    monkeypatch.setitem(config.MAX_TOKENS_PAR_ROLE, RoleLLM.AUDITEUR, 123)
    appeler_structure(ClientLLM(ModeCassette.REPLAY, tmp_path / "c"), ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert len(espion.requetes) == 1  # autre plafond : même cassette, rejouée


def cout_usd_reference(t: config.Tarif, entree: int, sortie: int) -> float:
    return (entree * t.entree_usd_par_mtok + sortie * t.sortie_usd_par_mtok) / 1e6


def test_rejeu_hors_budget(tmp_path: Path) -> None:
    _ecrire_cassette(tmp_path, RoleLLM.AUDITEUR, _messages(), _reponse('{"valeur": 3}'))
    ledger = RunLedger("run-x", tmp_path / "run-x", budget_usd=1e-9)
    appeler_structure(ClientLLM(ModeCassette.REPLAY, tmp_path), ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert ledger.depense_reelle_usd() == 0


# --------------------------------------------------------------------------- #
# Aucun en-tête HTTP ni secret dans les fichiers écrits
# --------------------------------------------------------------------------- #


def test_aucun_entete_http_dans_cassettes_et_reponses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "cle-secrete-de-test-0123456789"
    monkeypatch.setenv(config.ENV_OPENROUTER_CLE, secret)
    espion = _TransportEspion([_reponse('{"valeur": 3}')])
    client = ClientLLM(ModeCassette.RECORD, tmp_path / "cassettes", http=espion.client())
    appeler_structure(client, RunLedger("run-x", tmp_path / "run-x"), RoleLLM.AUDITEUR, _messages(), _Sortie)
    champs = set(Cassette.model_fields)
    assert champs == {
        "cle", "modele", "date_enregistrement", "nom_schema", "messages", "latency_ms", "reponse_brute", "fournisseur",
    }  # fmt: skip
    fichiers = [*(tmp_path / "cassettes").iterdir(), *(tmp_path / "run-x" / "reponses_llm").iterdir()]
    assert len(fichiers) == 2
    for fichier in fichiers:
        texte = fichier.read_text(encoding="utf-8")
        assert set(json.loads(texte)) == champs
        for interdit in (secret, "Bearer", "authorization", "Authorization", "headers"):
            assert interdit not in texte, (fichier.name, interdit)


def test_fixtures_http_sans_entetes_de_requete() -> None:
    racine = Path(__file__).parent / "fixtures"
    for fichier in (racine / "http").rglob("*.json"):
        assert set(json.loads(fichier.read_text(encoding="utf-8"))) <= {
            "methode", "url", "status", "content_type", "location", "corps_b64",
        }, fichier  # fmt: skip
    for fichier in (racine / "cassettes").rglob("*.json"):
        assert set(json.loads(fichier.read_text(encoding="utf-8"))) <= set(Cassette.model_fields), fichier


def test_usage_absent_jamais_estime(tmp_path: Path) -> None:
    _ecrire_cassette(tmp_path, RoleLLM.AUDITEUR, _messages(), {"choices": [{"message": {"content": '{"valeur": 1}'}}]})
    client = ClientLLM(ModeCassette.REPLAY, tmp_path)
    ledger = RunLedger("run-x", tmp_path / "run-x")
    with pytest.raises(UsageManquantError):
        appeler_structure(client, ledger, RoleLLM.AUDITEUR, _messages(), _Sortie)
    assert ledger.entrees == ()


# --------------------------------------------------------------------------- #
# Machines à états
# --------------------------------------------------------------------------- #


# Tables attendues recopiées à la main de la spec (section 4), volontairement
# indépendantes de schemas.py : le test compare l'implémentation à la spec.
STATUTS_CLAIM_SPEC = {
    "EXTRACTED", "REJECTED_VERBATIM", "REJECTED_PREFILTER", "REJECTED_SEMANTIC",
    "CLAIMED_UNPROVEN", "ESTABLISHED", "CONTRADICTED",
}  # fmt: skip
STATUTS_PREUVE_SPEC = {"CANDIDATE", "REJECTED_VERBATIM", "REJECTED_PREFILTER", "REJECTED_SEMANTIC", "CONFIRMED"}
PORTES_SPEC = {"extracteur", "verify", "prefilter", "auditeur", "contradicteur"}

# (départ, arrivée) -> portes autorisées ; départ None = création.
ATTENDU_CLAIM: dict[tuple[str | None, str], set[str]] = {
    (None, "EXTRACTED"): {"extracteur"},
    ("EXTRACTED", "REJECTED_VERBATIM"): {"verify"},
    ("EXTRACTED", "REJECTED_PREFILTER"): {"prefilter"},
    ("EXTRACTED", "REJECTED_SEMANTIC"): {"auditeur"},
    ("EXTRACTED", "CLAIMED_UNPROVEN"): {"auditeur"},
    ("EXTRACTED", "ESTABLISHED"): {"auditeur"},
    ("ESTABLISHED", "CONTRADICTED"): {"contradicteur"},
}
ATTENDU_PREUVE: dict[tuple[str | None, str], set[str]] = {
    (None, "CANDIDATE"): {"contradicteur"},
    ("CANDIDATE", "REJECTED_VERBATIM"): {"verify"},
    ("CANDIDATE", "REJECTED_PREFILTER"): {"prefilter"},
    ("CANDIDATE", "REJECTED_SEMANTIC"): {"auditeur"},
    ("CANDIDATE", "CONFIRMED"): {"auditeur"},
}


def test_enums_conformes_a_la_spec() -> None:
    assert {s.value for s in StatutClaim} == STATUTS_CLAIM_SPEC
    assert {s.value for s in StatutPreuve} == STATUTS_PREUVE_SPEC
    assert {p.value for p in Porte} == PORTES_SPEC


_OCCURRENCE = (Occurrence(offsets=Offsets(debut=0, fin=60), context_window="fenêtre"),)


def _ancre(c: Claim) -> Claim:
    return c.avec_ancrage(_OCCURRENCE, 1).avec_drapeaux(Drapeaux(is_future_or_protocol=False))


def _claim_en(statut: StatutClaim) -> Claim:
    extrait = Claim.extrait("c1", "pmid:1", "x" * 60)
    if statut is StatutClaim.EXTRACTED:
        return _ancre(extrait)
    if statut is StatutClaim.REJECTED_VERBATIM:
        return extrait.transitionner(statut, Porte.VERIFY, "citation < 40 caractères")
    if statut is StatutClaim.CONTRADICTED:
        etabli = _ancre(extrait).transitionner(StatutClaim.ESTABLISHED, Porte.AUDITEUR)
        return etabli.transitionner(statut, Porte.CONTRADICTEUR, "p1")
    (porte,) = ATTENDU_CLAIM[("EXTRACTED", statut.value)]
    return _ancre(extrait).transitionner(statut, Porte(porte), "motif")


def _preuve_en(statut: StatutPreuve) -> PreuveAdverse:
    candidate = PreuveAdverse.candidate("p1", "c1", "rapamycin lifespan humans", "pmid:2", "y" * 60)
    candidate = candidate.avec_ancrage(_OCCURRENCE, 1).avec_drapeaux(
        Drapeaux(is_future_or_protocol=False)
    )
    if statut is StatutPreuve.CANDIDATE:
        return candidate
    (porte,) = ATTENDU_PREUVE[("CANDIDATE", statut.value)]
    return candidate.transitionner(statut, Porte(porte), "motif")


@pytest.mark.parametrize("de", sorted(STATUTS_CLAIM_SPEC))
@pytest.mark.parametrize("vers", sorted(STATUTS_CLAIM_SPEC))
@pytest.mark.parametrize("porte", sorted(PORTES_SPEC))
def test_transitions_claim_exhaustives(de: str, vers: str, porte: str) -> None:
    claim = _claim_en(StatutClaim(de))
    if porte in ATTENDU_CLAIM.get((de, vers), set()):
        suivant = claim.transitionner(StatutClaim(vers), Porte(porte), "motif")
        assert suivant.statut.value == vers
        assert suivant.historique[-1].porte.value == porte
        assert claim.statut.value == de  # l'original est inchangé
    else:
        with pytest.raises(InvalidTransitionError):
            claim.transitionner(StatutClaim(vers), Porte(porte), "motif")


@pytest.mark.parametrize("de", sorted(STATUTS_PREUVE_SPEC))
@pytest.mark.parametrize("vers", sorted(STATUTS_PREUVE_SPEC))
@pytest.mark.parametrize("porte", sorted(PORTES_SPEC))
def test_transitions_preuve_exhaustives(de: str, vers: str, porte: str) -> None:
    preuve = _preuve_en(StatutPreuve(de))
    if porte in ATTENDU_PREUVE.get((de, vers), set()):
        assert preuve.transitionner(StatutPreuve(vers), Porte(porte), "motif").statut_preuve.value == vers
    else:
        with pytest.raises(InvalidTransitionError):
            preuve.transitionner(StatutPreuve(vers), Porte(porte), "motif")


@pytest.mark.parametrize("vers", sorted(STATUTS_CLAIM_SPEC))
@pytest.mark.parametrize("porte", sorted(PORTES_SPEC))
def test_creation_claim_conforme(vers: str, porte: str) -> None:
    """Création (départ None) : seul l'extracteur crée, et seulement en EXTRACTED."""
    historique = [{"de": None, "vers": vers, "porte": porte}]
    donnees = {"claim_id": "c1", "source_id": "pmid:1", "citation_verbatim": "x", "statut": vers,
               "historique": historique, "motif_rejet": "m" if vers.startswith("REJECTED") else None}  # fmt: skip
    if porte in ATTENDU_CLAIM.get((None, vers), set()):
        assert Claim.model_validate(donnees).statut.value == vers
    else:
        with pytest.raises(InvalidTransitionError):
            Claim.model_validate(donnees)


def test_prefiltre_ne_peut_pas_rendre_claimed_unproven() -> None:
    with pytest.raises(InvalidTransitionError):
        _claim_en(StatutClaim.EXTRACTED).transitionner(StatutClaim.CLAIMED_UNPROVEN, Porte.PREFILTER)


def test_json_falsifie_rejete_a_la_relecture() -> None:
    etabli = _claim_en(StatutClaim.ESTABLISHED).model_dump(mode="json")

    saut = {**etabli, "statut": "CONTRADICTED"}
    with pytest.raises(InvalidTransitionError):
        Claim.model_validate(saut)

    porte_usurpee = json.loads(json.dumps(etabli))
    porte_usurpee["historique"][-1]["porte"] = "prefilter"
    with pytest.raises(InvalidTransitionError):
        Claim.model_validate(porte_usurpee)

    assert Claim.model_validate(etabli) == _claim_en(StatutClaim.ESTABLISHED)


def test_statut_post_prefiltre_exige_ancrage_et_drapeaux() -> None:
    extrait = Claim.extrait("c1", "pmid:1", "x" * 60)
    with pytest.raises(ValidationError, match="verify.py et prefilter.py"):
        extrait.transitionner(StatutClaim.ESTABLISHED, Porte.AUDITEUR)
    with pytest.raises(ValidationError, match="sans ancrage"):
        extrait.avec_drapeaux(Drapeaux(is_future_or_protocol=True))


def test_occurrences_coherentes() -> None:
    occ = [Occurrence(offsets=Offsets(debut=10 * k, fin=10 * k + 5), context_window=f"f{k}") for k in range(3)]
    extrait = Claim.extrait("c1", "pmid:1", "x" * 60)
    ancre = extrait.avec_ancrage(tuple(occ), 7)
    assert ancre.fenetres == ("f0", "f1", "f2") and ancre.nombre_occurrences == 7
    with pytest.raises(ValidationError, match="incohérentes"):
        extrait.avec_ancrage(tuple(occ[:2]), 7)  # 7 trouvées : 3 doivent être conservées
    with pytest.raises(ValidationError):
        extrait.avec_ancrage((*occ, occ[0]), 4)  # au plus 3


def test_rejet_exige_un_motif() -> None:
    with pytest.raises(ValidationError, match="motif_rejet"):
        Claim.extrait("c1", "pmid:1", "court").transitionner(StatutClaim.REJECTED_VERBATIM, Porte.VERIFY)


def test_citation_invalide_n_est_pas_une_erreur_de_parsing() -> None:
    # Une citation vide ou un ID inventé doivent atteindre verify.py, pas casser le parsing.
    claim = Claim.extrait("c1", "pmid:inventé", "")
    assert claim.statut is StatutClaim.EXTRACTED


def test_modeles_geles() -> None:
    claim = _claim_en(StatutClaim.EXTRACTED)
    with pytest.raises(ValidationError):
        claim.statut = StatutClaim.ESTABLISHED  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Claim.model_validate({**claim.model_dump(), "champ_inconnu": 1})


# --------------------------------------------------------------------------- #
# VerdictRoutage et CommandeD1
# --------------------------------------------------------------------------- #


def test_verdict_routage_coherent() -> None:
    VerdictRoutage(verdict=Verdict.REFUS, motif_refus=MotifRefus.PAS_D_EQUATION)
    with pytest.raises(ValidationError):
        VerdictRoutage(verdict=Verdict.REFUS)
    with pytest.raises(ValidationError):
        VerdictRoutage(verdict=Verdict.SIMULABLE, papier_cible="pmid:1")
    with pytest.raises(ValidationError):
        VerdictRoutage(verdict=Verdict.REFUS, motif_refus=MotifRefus.JUGE_REFUSE, papier_cible="pmid:1")


def _commande(**surcharges: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "commande_id": "mm-1",
        "piste_id": "enzymo",
        "sujet_id": "michaelis-menten",
        "origine": "manuel",
        "source_id": "manuel",
        "citation_equation": r"v(S) = \frac{V_{\max} \, S}{K_m + S}",
        "convention": "S et Km dans la même unité de concentration ; v dans l'unité de Vmax.",
        "signature_fonction": {"nom": "michaelis_menten", "entrees": ["S", "Vmax", "Km"], "sorties": ["v"]},
        "description_pour_forgeron": "Vitesse initiale de Michaelis-Menten.",
        "parametres": {},
        "epsilon_abs": 1e-12,
        "epsilon_rel": 1e-9,
        "cas_publics": [{"entrees": {"S": 1.0, "Vmax": 2.0, "Km": 1.0}, "attendu": {"v": 1.0}}],
        "cas_caches": [{"entrees": {"S": 0.0, "Vmax": 2.0, "Km": 1.0}, "attendu": {"v": 0.0}}],
        "graine_grille": 42,
        "domaine_grille": {
            "S": {"bas": 0, "haut": 1000},
            "Vmax": {"bas": 0.1, "haut": 100},
            "Km": {"bas": 0.01, "haut": 100},
        },
        "invariants": [
            {"type": "bounded", "sortie": "v", "min": 0, "max": "Vmax"},
            {"type": "monotonic", "sortie": "v", "entree": "S", "direction": "increasing"},
        ],
    }
    base.update(surcharges)
    return base


def test_commande_valide_et_signature_rendue() -> None:
    commande = CommandeD1.model_validate(_commande())
    assert commande.signature_fonction.rendre() == "michaelis_menten(S: float, Vmax: float, Km: float) -> float"
    assert len(commande.contrat_hash) == 64


def test_contrainte_ratio_max_declarative() -> None:
    contrainte = {"type": "ratio_max", "num": "Vmax", "den": "Km", "max": 20}
    commande = CommandeD1.model_validate(_commande(contraintes_grille=[contrainte]))
    assert commande.model_dump(mode="json")["contraintes_grille"] == [{**contrainte, "max": 20.0}]


def test_contrat_hash_stable_et_sensible() -> None:
    a = CommandeD1.model_validate(_commande())
    b = CommandeD1.model_validate(dict(reversed(list(_commande().items()))))  # ordre des clés inversé
    assert a.contrat_hash == b.contrat_hash
    assert CommandeD1.model_validate(_commande(graine_grille=43)).contrat_hash != a.contrat_hash
    assert CommandeD1.model_validate(_commande(epsilon_rel=1e-8)).contrat_hash != a.contrat_hash


def test_commande_profondement_immuable() -> None:
    commande = CommandeD1.model_validate(_commande())
    with pytest.raises(TypeError):
        commande.cas_caches[0].entrees["S"] = 5.0  # type: ignore[index]
    with pytest.raises(TypeError):
        commande.domaine_grille["S"] = commande.domaine_grille["Km"]  # type: ignore[index]


@pytest.mark.parametrize(
    "surcharges",
    [
        {"invariants": [{"type": "bounded", "sortie": "w", "min": 0, "max": None}]},
        {"invariants": [{"type": "bounded", "sortie": "v", "min": "Vinconnu", "max": None}]},
        {"invariants": [{"type": "bounded", "sortie": "v", "min": None, "max": None}]},
        {"invariants": [{"type": "monotonic", "sortie": "v", "entree": "t", "direction": "increasing"}]},
        {"invariants": [{"type": "conservation_sum", "sorties": ["v", "v"], "cible": 1, "tol": 0}]},
        {"invariants": [{"type": "finite", "sortie": "v"}]},
        {"cas_caches": [{"entrees": {"S": 0.0, "Vmax": 2.0}, "attendu": {"v": 0.0}}]},
        {"cas_publics": [{"entrees": {"S": 1.0, "Vmax": 2.0, "Km": 1.0}, "attendu": {"v": math.nan}}]},
        {"domaine_grille": {"S": {"bas": 0, "haut": 1}}},
        {"origine": "pont_d0"},
        {"run_id_d0": "run-1"},
        {"signature_fonction": {"nom": "f", "entrees": ["__class__"], "sorties": ["v"]}},
        {"signature_fonction": {"nom": "f", "entrees": ["S", "S"], "sorties": ["v"]}},
        {"sujet_id": "../evasion"},
        {"cas_caches": []},
        {"contraintes_grille": [{"type": "ratio_max", "num": "Vmax/Km", "den": "Km", "max": 20}]},
        {"contraintes_grille": [{"type": "ratio_max", "num": "Vmax", "den": "S", "max": 20}]},  # S.bas = 0
        {"contraintes_grille": [{"type": "ratio_max", "num": "Vmax", "den": "Km", "max": 0}]},
        {"contraintes_grille": [{"type": "expression", "expr": "Vmax / Km <= 20"}]},
        {"contraintes_grille": ["Vmax / Km <= 20"]},
        {"contraintes_grille": [{"num": "Vmax", "den": "Km", "max": 20}]},
    ],
)
def test_commande_invalide_refusee(surcharges: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CommandeD1.model_validate(_commande(**surcharges))
