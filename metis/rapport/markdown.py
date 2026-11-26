"""Rapport Markdown d'un run, généré uniquement depuis ses fichiers (0 LLM).

Aucune horloge, aucun ordre de dictionnaire implicite : deux générations sur les mêmes
fichiers donnent le même octet.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from metis.schemas import (
    STATUTS_REJET_CLAIM,
    Claim,
    CoutsRun,
    Incident,
    PreuveAdverse,
    QuestionD0,
    RequeteJournal,
    StatutClaim,
    StatutPreuve,
    VerdictRoutage,
)
from metis.storage import ecrire_texte_atomique, lire_json


def _lien(source_id: str) -> str:
    if source_id.startswith("pmid:"):
        pmid = source_id.removeprefix("pmid:")
        return f"[{source_id}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
    if source_id.startswith("pmc:"):
        return f"[{source_id}](https://pmc.ncbi.nlm.nih.gov/articles/{source_id.removeprefix('pmc:')}/)"
    if source_id.startswith("arxiv:"):
        identifiant = source_id.removeprefix("arxiv:").removesuffix(":tex")
        return f"[{source_id}](https://arxiv.org/abs/{identifiant})"
    return f"`{_cellule(source_id)}`"


def _cellule(texte: str | None) -> str:
    return " ".join((texte or "").split()).replace("|", "\\|")


def _lire_liste(dossier: Path, nom: str) -> list[object]:
    chemin = dossier / nom
    if not chemin.exists():
        return []
    donnees = lire_json(chemin)
    assert isinstance(donnees, list), nom
    return donnees


def generer_rapport(dossier: Path) -> str:
    question = QuestionD0.model_validate(lire_json(dossier / "question.json"))
    verdict = (
        VerdictRoutage.model_validate(lire_json(dossier / "verdict.json"))
        if (dossier / "verdict.json").exists()
        else None
    )
    claims = [Claim.model_validate(c) for c in _lire_liste(dossier, "claims.json")]
    preuves = [PreuveAdverse.model_validate(p) for p in _lire_liste(dossier, "preuves_adverses.json")]
    requetes = [RequeteJournal.model_validate(r) for r in _lire_liste(dossier, "requetes.json")]
    incidents = [Incident.model_validate(i) for i in _lire_liste(dossier, "incidents.json")]
    couts = CoutsRun.model_validate(lire_json(dossier / "couts.json")) if (dossier / "couts.json").exists() else None

    lignes: list[str] = [f"# Rapport Mêtis D0 — {question.question_id}", ""]

    lignes += [
        "## 1. Question",
        "",
        f"- **Sous-question** : {question.sous_question}",
        f"- **Espèce cible** : `{question.espece_cible.value}`",
        f"- **Mesurande** : {question.mesurande.nom} ({question.mesurande.unite})",
        f"- **Exclusions** : {', '.join(question.exclusions) if question.exclusions else 'aucune'}",
        "",
    ]

    lignes += ["## 2. Verdict exécutif", ""]
    if verdict is None:
        lignes.append("Aucun verdict : le run n'est pas allé jusqu'au routeur.")
    elif verdict.motif_refus is None:
        assert verdict.observable is not None
        lignes.append(
            f"**simulable** — équation citée dans {_lien(verdict.papier_cible or '')}, observable "
            f"« {verdict.observable.nom} » ({verdict.observable.unite})."
        )
        lignes += ["", f"> {_cellule(verdict.citation_equation)}"]
    else:
        lignes.append(f"**refus** — motif `{verdict.motif_refus.value}`.")
    lignes.append("")

    etablis = [c for c in claims if c.statut is StatutClaim.ESTABLISHED]
    lignes += ["## 3. Claims ESTABLISHED survivants", ""]
    if etablis:
        lignes += ["| claim | source | citation verbatim | statut |", "|---|---|---|---|"]
        lignes += [
            f"| {c.claim_id} | {_lien(c.source_id)} | {_cellule(c.citation_verbatim)} | {c.statut.value} |"
            for c in etablis
        ]
    else:
        lignes.append("Aucun.")
    lignes.append("")

    par_preuve = {p.preuve_id: p for p in preuves}
    contredits = [c for c in claims if c.statut is StatutClaim.CONTRADICTED]
    lignes += ["## 4. Claims CONTRADICTED", ""]
    if contredits:
        lignes += ["| claim | citation | preuve adverse confirmée |", "|---|---|---|"]
        for c in contredits:
            preuve = par_preuve.get(c.historique[-1].motif or "")
            adverse = f"{_lien(preuve.source_id)} : {_cellule(preuve.citation_verbatim)}" if preuve else "?"
            lignes.append(f"| {c.claim_id} ({_lien(c.source_id)}) | {_cellule(c.citation_verbatim)} | {adverse} |")
    else:
        lignes.append("Aucun.")
    lignes.append("")

    morts_claims = [c for c in claims if c.statut in STATUTS_REJET_CLAIM or c.statut is StatutClaim.CLAIMED_UNPROVEN]
    morts_preuves = [p for p in preuves if p.statut_preuve is not StatutPreuve.CONFIRMED]
    lignes += ["## 5. Cimetière", ""]
    if morts_claims or morts_preuves:
        lignes += ["| élément | source | citation | statut | porte | motif |", "|---|---|---|---|---|---|"]
        for c in morts_claims:
            derniere = c.historique[-1]
            lignes.append(
                f"| {c.claim_id} | {_lien(c.source_id)} | {_cellule(c.citation_verbatim)} | {c.statut.value} "
                f"| {derniere.porte.value} | {_cellule(derniere.motif)} |"
            )
        for p in morts_preuves:
            fin = p.historique[-1]
            lignes.append(
                f"| {p.preuve_id} → {p.claim_cible_id} | {_lien(p.source_id)} | {_cellule(p.citation_verbatim)} "
                f"| {p.statut_preuve.value} | {fin.porte.value} | {_cellule(fin.motif)} |"
            )
    else:
        lignes.append("Vide.")
    if incidents:
        lignes += ["", "Incidents :", ""]
        lignes += [f"- `{i.etape}` sur `{_cellule(i.cible)}` : {_cellule(i.motif)}" for i in incidents]
    lignes.append("")

    adverses = [r for r in requetes if r.origine.value == "adverse"]
    lignes += ["## 6. Requêtes adverses", ""]
    if adverses:
        lignes += ["| claim visé | base | requête | résultats | erreur |", "|---|---|---|---|---|"]
        lignes += [
            f"| {r.claim_cible_id} | {r.base.value} | {_cellule(r.texte)} | {r.nb_resultats} | {_cellule(r.erreur)} |"
            for r in adverses
        ]
    else:
        lignes.append("Aucune (aucun claim ESTABLISHED à réfuter).")
    lignes.append("")

    lignes += ["## 7. Bilan financier", ""]
    if couts is None or not couts.entrees:
        lignes.append("Aucun appel LLM.")
    else:
        par_role: dict[str, float] = defaultdict(float)
        par_modele: dict[str, float] = defaultdict(float)
        rejoue = 0.0
        for e in couts.entrees:
            par_role[e.role.value] += e.cost_usd
            par_modele[e.model] += e.cost_usd
            if e.cassette:
                rejoue += e.cost_usd
        total = couts.cout_total_usd
        part = rejoue / total if total else 0.0
        lignes += [
            f"- **Coût total** : {total:.4f} USD sur {len(couts.entrees)} appels",
            f"- **Part rejouée depuis cassettes** : {rejoue:.4f} USD ({part:.0%})",
            "",
            "| rôle | USD |",
            "|---|---|",
        ]
        lignes += [f"| {r} | {par_role[r]:.4f} |" for r in sorted(par_role)]
        lignes += ["", "| modèle | USD |", "|---|---|"]
        lignes += [f"| {m} | {par_modele[m]:.4f} |" for m in sorted(par_modele)]
        refus: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for e in couts.entrees:
            refus[e.role.value][1] += 1
            refus[e.role.value][0] += e.refus_fournisseur
        lignes += ["", "| rôle | refus du fournisseur / appels |", "|---|---|"]
        lignes += [f"| {r} | {refus[r][0]} / {refus[r][1]} |" for r in sorted(refus)]
    lignes.append("")
    return "\n".join(lignes)


def ecrire_rapport(dossier: Path) -> str:
    rapport = generer_rapport(dossier)
    ecrire_texte_atomique(dossier / "rapport.md", rapport)
    return rapport
