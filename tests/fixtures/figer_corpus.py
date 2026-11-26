"""Fige des réponses HTTP réelles de PubMed dans tests/fixtures/http/<nom>/ (hors pytest).

Usage :
    .venv/bin/python tests/fixtures/figer_corpus.py notices <nom> <pmid> [<pmid> ...]
    .venv/bin/python tests/fixtures/figer_corpus.py arxiv <nom> <search_query> <max>

notices : seules les notices PubMed (titre + abstract), un document par PMID.
arxiv : résumés et sources LaTeX de la requête.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from http_fige import client_fige  # noqa: E402

from metis.literature.arxiv import CollecteurArxiv  # noqa: E402
from metis.literature.pubmed import CollecteurPubMed  # noqa: E402

DOSSIER_HTTP = Path(__file__).parent / "http"


def main(arguments: list[str]) -> None:
    if len(arguments) < 3 or arguments[0] not in ("notices", "arxiv"):
        raise SystemExit(__doc__)
    commande, nom = arguments[0], arguments[1]
    client = client_fige(DOSSIER_HTTP / nom, enregistrer=True)
    if commande == "notices":
        documents = [CollecteurPubMed(client).notice(pmid)[0] for pmid in arguments[2:]]
    else:
        documents = list(CollecteurArxiv(client).collecter(arguments[2], int(arguments[3])))
    for document in documents:
        print(f"{document.source_id}: {len(document.texte_nettoye)} caractères")


if __name__ == "__main__":
    main(sys.argv[1:])
