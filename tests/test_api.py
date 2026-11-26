"""Tests de l'API FastAPI (portes déterministes, démo rejouable et évaluations, 0 USD)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from metis.api import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "2.2.0"
    assert body["lexiques_version"] == "3"
    assert body["textclean_version"] == "2"


def test_verify_accepte_citation_verbatim_humaine() -> None:
    source = (
        "In this randomized controlled trial of 3270 healthy ambulatory women, "
        "supplementation with vitamin D3 and calcium reduces the risk of hip fractures (P = 0.043)."
    )
    citation = "supplementation with vitamin D3 and calcium reduces the risk of hip fractures (P = 0.043)."
    r = client.post(
        "/v1/verify",
        json={"source_id": "pmid:1331788", "source_text": source, "citation": citation},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["statut"] == "EXTRACTED"
    assert body["verbatim_ok"] is True
    assert body["occurrences"] == 1
    assert body["motif_rejet"] is None


def test_verify_rejette_citation_hallucinee() -> None:
    source = (
        "In this randomized controlled trial of 3270 healthy ambulatory women, "
        "supplementation with vitamin D3 and calcium reduces the risk of hip fractures."
    )
    citation = "Vitamin D3 and calcium completely cure osteoporosis in 100 percent of elderly women."
    r = client.post(
        "/v1/verify",
        json={"source_id": "pmid:1331788", "source_text": source, "citation": citation},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["statut"] == "REJECTED_VERBATIM"
    assert body["verbatim_ok"] is False
    assert body["occurrences"] == 0
    assert body["motif_rejet"] is not None


def test_verify_rejette_transposition_animale_au_prefiltre() -> None:
    source = (
        "In a second study, female mice were pair-fed 30% EtOH diets with or without "
        "dietary vitamin D supplementation for eight weeks."
    )
    citation = "In a second study, female mice were pair-fed 30% EtOH diets with or without"
    r = client.post(
        "/v1/verify",
        json={
            "source_id": "pmid:22892342",
            "source_text": source,
            "citation": citation,
            "espece_cible": "human_in_vivo",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["statut"] == "REJECTED_PREFILTER"
    assert body["verbatim_ok"] is True
    assert body["motif_rejet"] == "espece_exclue_sans_marqueur_humain: mice"


def test_audit_demo_replay_sans_reseau() -> None:
    r = client.post("/v1/audit/demo")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "replay"
    assert body["cout_reel_usd"] == 0.0
    assert body["nb_appels_llm_rejoues"] > 0
    assert body["nb_documents"] > 0
    assert body["verdict"] is not None
    assert "## 1. Question" in body["rapport_markdown"]
    assert "## 7. Bilan financier" in body["rapport_markdown"]


def test_evals_expose_opus_et_qwen() -> None:
    r = client.get("/v1/evals")
    assert r.status_code == 200
    rapports = r.json()["rapports"]
    modeles = {item["modele"] for item in rapports}
    assert "anthropic/claude-opus-5.5" in modeles
    assert "qwen/qwen3.8-max-0902" in modeles
