"""Appel de sondage unique (~0,01 USD) : l'auditeur sur le contrôle positif etabli_chapuy.

Usage :
    OPENROUTER_API_KEY=... .venv/bin/python tests/fixtures/sonder.py

La requête est exactement celle de test_red_team_complet : en mode record_missing, la
cassette écrite ici est rejouée ensuite par le test (aucun second appel facturé).
Budget du run : 0,40 USD, car la réservation porte sur le coût maximal possible
(16 000 tokens de sortie) ; le coût réel d'un appel auditeur a été de 0,010 à 0,012 USD
lors de l'enregistrement du 2026-11-03.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["METIS_CASSETTE"] = "record_missing"

import test_red_team_critic as red_team  # noqa: E402

from metis.llm import RunLedger  # noqa: E402
from metis.pipeline.auditeur import auditer_claim  # noqa: E402


def main() -> None:
    etat, deps = red_team._etat_filtre(Path(tempfile.mkdtemp(prefix="sonde-")), red_team.QUESTION)
    ledger = RunLedger("sonde", deps.dossier_run, budget_usd=0.40)
    claim = red_team._par_nom(etat, red_team.QUESTION)["etabli_chapuy"]
    juge = auditer_claim(claim, red_team.QUESTION, etat.corpus[claim.source_id], deps.client, ledger)
    (entree,) = ledger.entrees
    reponse = json.loads((deps.dossier_run / (entree.fichier_reponse or "")).read_text(encoding="utf-8"))
    choix = reponse["reponse_brute"]["choices"][0]
    print(f"modèle          : {entree.model}")
    print(f"fournisseur     : {entree.fournisseur}")
    print(f"rejeu cassette  : {entree.cassette}")
    print(f"finish_reason   : {choix.get('finish_reason')} / {choix.get('native_finish_reason')}")
    print(f"refus           : {entree.refus_fournisseur}")
    print(f"coût            : {entree.cost_usd:.6f} USD ({entree.prompt_tokens} / {entree.completion_tokens} tokens)")
    print(f"statut          : {juge.statut.value}")
    print(f"motif           : {juge.historique[-1].motif}")


if __name__ == "__main__":
    main()
