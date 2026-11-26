"""Tests du support des modèles locaux (Ollama / serveur compatible OpenAI) sans réseau."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from metis import config
from metis.config import ModeCassette, Tarif
from metis.llm import (
    CleApiManquanteError,
    ClientLLM,
    RunLedger,
    UnknownModelPricingError,
    tarif,
)
from metis.schemas import Message, RoleLLM


class _ReponseTest(BaseModel):
    ok: bool


def test_tarif_modele_local_gratuit_et_inconnu_refuse() -> None:
    assert tarif("ollama/qwen2.5:7b") == Tarif(0.0, 0.0)
    assert tarif("local/llama3.1:8b") == Tarif(0.0, 0.0)
    with pytest.raises(UnknownModelPricingError):
        tarif("inconnu/modele-payant")


def test_modele_local_sur_hote_distant_exige_toujours_cle_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(config.ENV_OPENROUTER_CLE, raising=False)
    monkeypatch.setenv(config.ENV_MODELE_JUGE, "ollama/qwen2.5:7b")
    monkeypatch.setenv(config.ENV_LLM_BASE_URL, "https://openrouter.ai/api/v1/chat/completions")

    client = ClientLLM(ModeCassette.LIVE, tmp_path / "cassettes")
    ledger = RunLedger("test-local-fail-closed", tmp_path / "run")

    with pytest.raises(CleApiManquanteError):
        client.completer(
            ledger,
            RoleLLM.AUDITEUR,
            (Message(role="user", content="test"),),
            _ReponseTest,
        )


def test_modele_local_loopback_sans_cle_api_et_cout_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(config.ENV_OPENROUTER_CLE, raising=False)
    monkeypatch.setenv(config.ENV_MODELE_JUGE, "ollama/qwen2.5:7b")
    monkeypatch.setenv(config.ENV_LLM_BASE_URL, "http://127.0.0.1:11434/v1/chat/completions")
    monkeypatch.setenv("METIS_PROVIDER_AUDITEUR", "anthropic")

    recu: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        recu["url"] = str(request.url)
        recu["auth"] = request.headers.get("authorization")
        recu["corps"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7},
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(_handler))
    try:
        client = ClientLLM(ModeCassette.LIVE, tmp_path / "cassettes", http=http)
        ledger = RunLedger("test-ollama-local", tmp_path / "run")
        contenu = client.completer(
            ledger,
            RoleLLM.AUDITEUR,
            (Message(role="user", content="ping"),),
            _ReponseTest,
        )
    finally:
        http.close()

    assert contenu == '{"ok": true}'
    assert recu["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert recu["auth"] is None
    corps = recu["corps"]
    assert isinstance(corps, dict)
    assert corps["model"] == "qwen2.5:7b"  # préfixe ollama/ retiré pour le serveur local
    assert "provider" not in corps  # champ spécifique OpenRouter absent en local
    assert ledger.depense_reelle_usd() == 0.0
    assert len(ledger.entrees) == 1
    assert ledger.entrees[0].cost_usd == 0.0
