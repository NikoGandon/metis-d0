"""Nettoyage des sources : PMC/PubMed XML et LaTeX arXiv -> texte (0 LLM).

Nettoyage purement structurel, appliqué au texte source uniquement :
- PMC : texte des balises conservé, <sup>/<sub> aplatis, seuls les appels
  <xref ref-type="bibr|fn"> sont retirés (avec les crochets ou parenthèses qui les
  entourent exactement) ; aucun motif textuel du type [12] n'est retiré, pour ne
  jamais toucher [0, 1], [S], [Ca2+] ou [3H]thymidine ;
- PubMed et arXiv : rien n'est retiré (espaces seulement ; commentaires LaTeX).

Aucune suppression textuelle ne subsistant, la seule fonction appliquée à
l'identique au texte source et à la citation est verify.normaliser. Toute
modification de ce module incrémente config.TEXTCLEAN_VERSION, enregistrée dans
chaque Document.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from metis import config
from metis.schemas import Document, FormatSource

# Commentaire LaTeX : % non échappé jusqu'à la fin de ligne (newline conservé)
_RE_COMMENTAIRE_LATEX = re.compile(r"(?<!\\)%[^\n]*")
_RE_ESPACES = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# XML (JATS PMC, PubMed efetch)
# --------------------------------------------------------------------------- #

# Éléments dont le contenu est séparé de ses voisins par une espace.
_BLOCS_INLINE = frozenset({"p", "title", "label", "caption", "td", "th", "list-item", "AbstractText"})
# Éléments traités comme blocs séparés quand ils apparaissent dans un <p>.
_BLOCS_EXTRAITS = frozenset({"table-wrap", "fig"})
_RE_PONCTUATION_SEULE = re.compile(r"^[\s,;–—\-]*$")


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _formule(element: ET.Element) -> str:
    """Formule : source TeX si disponible (<tex-math>), sinon texte MathML aplati."""
    for descendant in element.iter():
        if _local(descendant.tag) == "tex-math":
            return "".join(descendant.itertext())
    return " ".join("".join(element.itertext()).split())


def _est_appel(element: ET.Element) -> bool:
    """<xref> bibliographique ou de note. Les renvois internes (table, fig, sec) gardent
    leur texte, qui fait partie de la phrase (« see Table 1 »)."""
    return _local(element.tag) == "xref" and element.get("ref-type") in ("bibr", "fn")


def _que_des_appels(element: ET.Element) -> bool:
    """<sup>/<sub> ne contenant que des appels et de la ponctuation (appels en exposant)."""
    enfants = list(element)
    if not enfants or not all(_est_appel(e) for e in enfants):
        return False
    reste = (element.text or "") + "".join(e.tail or "" for e in enfants)
    return bool(_RE_PONCTUATION_SEULE.match(reste))


_FERMANTES = {"[": "]", "(": ")"}


def _retirer_appel(avant: str, apres: str) -> tuple[str, str]:
    """Retire un groupe d'appels entre avant et apres.

    Les délimiteurs qui l'entourent exactement (« [ » … « ] », « ( » … « ) ») partent
    avec lui ; l'espace qui le précède aussi, sauf s'il est collé au mot suivant.
    """
    gauche, droite = avant.rstrip(), apres.lstrip()
    if gauche and gauche[-1] in _FERMANTES and droite.startswith(_FERMANTES[gauche[-1]]):
        gauche, apres = gauche[:-1].rstrip(), droite[1:]
    if not apres or apres[0].isspace() or apres[0] in ".,;:!?)]":
        return gauche, apres
    return avant, apres


def _inline(element: ET.Element) -> str:
    texte = element.text or ""
    enfants = list(element)
    i = 0
    while i < len(enfants):
        enfant = enfants[i]
        nom = _local(enfant.tag)
        if _est_appel(enfant) or (nom in ("sup", "sub") and _que_des_appels(enfant)):
            # Groupe d'appels consécutifs séparés par de la ponctuation : [1, 3–5], (A; B)
            while (
                i + 1 < len(enfants)
                and _est_appel(enfants[i + 1])
                and _RE_PONCTUATION_SEULE.match(enfants[i].tail or "")
            ):
                i += 1
            texte, queue = _retirer_appel(texte, enfants[i].tail or "")
            texte += queue
            i += 1
            continue
        if nom in _BLOCS_EXTRAITS:
            pass
        elif nom in ("inline-formula", "disp-formula"):
            texte += _formule(enfant)
        elif nom in _BLOCS_INLINE:
            texte += f" {_inline(enfant)} "
        else:
            texte += _inline(enfant)
        texte += enfant.tail or ""
        i += 1
    return texte


def _bloc(texte: str) -> str:
    return _RE_ESPACES.sub(" ", texte).strip()


def _ajouter(blocs: list[str], etiquette: str, texte: str) -> None:
    contenu = _bloc(texte)
    if contenu or etiquette:
        blocs.append(f"{etiquette} {contenu}".strip() if etiquette else contenu)


def _parcourir(element: ET.Element, blocs: list[str]) -> None:
    nom = _local(element.tag)
    if nom in ("ref-list", "fn-group", "back"):
        return
    if nom == "sec":
        for enfant in element:
            if _local(enfant.tag) == "title":
                _ajouter(blocs, "[SECTION]", _inline(enfant))
            else:
                _parcourir(enfant, blocs)
    elif nom == "p":
        _ajouter(blocs, "", _inline(element))
        for enfant in element.iter():
            if enfant is not element and _local(enfant.tag) in _BLOCS_EXTRAITS:
                _parcourir(enfant, blocs)
    elif nom == "table-wrap":
        entete = " ".join(_inline(e) for e in element if _local(e.tag) in ("label", "caption"))
        _ajouter(blocs, "[TABLEAU]", entete)
        for ligne in element.iter():
            if _local(ligne.tag) == "tr":
                cellules = [_bloc(_inline(c)) for c in ligne if _local(c.tag) in ("td", "th")]
                blocs.append(" | ".join(cellules))
        for pied in element.iter():
            if _local(pied.tag) == "table-wrap-foot":
                _ajouter(blocs, "[NOTE DE TABLEAU]", _inline(pied))
    elif nom == "fig":
        legende = " ".join(_inline(e) for e in element if _local(e.tag) in ("label", "caption"))
        _ajouter(blocs, "[LÉGENDE]", legende)
    elif nom == "disp-formula":
        blocs.append(f"[ÉQUATION] {_formule(element).strip()}")
    else:
        for enfant in element:
            _parcourir(enfant, blocs)


def _unique(racine: ET.Element, nom: str) -> ET.Element:
    trouves = [e for e in racine.iter() if _local(e.tag) == nom]
    if len(trouves) != 1:
        raise ValueError(f"attendu exactement un <{nom}>, trouvé {len(trouves)}")
    return trouves[0]


def nettoyer_pmc_xml(xml: str) -> str:
    """Texte intégral JATS -> blocs étiquetés séparés par une ligne vide.

    Texte des balises conservé (<i>in vivo</i> -> in vivo), <sup>/<sub> aplatis,
    seuls les appels <xref ref-type="bibr|fn"> supprimés, tableaux et légendes conservés
    dans des blocs [TABLEAU] / [LÉGENDE]. Les références (<back>, <ref-list>) sont exclues.
    """
    article = _unique(ET.fromstring(xml), "article")
    blocs: list[str] = []
    for front in (e for e in article if _local(e.tag) == "front"):
        for titre in (e for e in front.iter() if _local(e.tag) == "article-title"):
            _ajouter(blocs, "[TITRE]", _inline(titre))
            break
        for resume in (e for e in front.iter() if _local(e.tag) == "abstract"):
            blocs.append("[RÉSUMÉ]")
            for enfant in resume:
                _parcourir(enfant, blocs)
    for corps in (e for e in article if _local(e.tag) == "body"):
        for enfant in corps:
            _parcourir(enfant, blocs)
    return "\n\n".join(blocs)


def nettoyer_pubmed_xml(xml: str) -> str:
    """Notice efetch PubMed (un seul PubmedArticle) -> titre + abstract étiqueté."""
    article = _unique(ET.fromstring(xml), "PubmedArticle")
    blocs: list[str] = []
    titre = _unique(article, "ArticleTitle")
    _ajouter(blocs, "[TITRE]", _inline(titre))
    blocs.append("[RÉSUMÉ]")
    for partie in (e for e in article.iter() if _local(e.tag) == "AbstractText"):
        etiquette = partie.get("Label")
        _ajouter(blocs, f"{etiquette}:" if etiquette else "", _inline(partie))
    return "\n\n".join(blocs)


# --------------------------------------------------------------------------- #
# LaTeX et résumés arXiv
# --------------------------------------------------------------------------- #


def nettoyer_latex(source: str) -> str:
    """Commentaires supprimés ; tout le reste (maths, \\cite) conservé tel quel."""
    return _RE_COMMENTAIRE_LATEX.sub("", source)


_ATOM = "{http://www.w3.org/2005/Atom}"


def nettoyer_resume_arxiv(entree_atom: str) -> str:
    """Entrée Atom de l'API arXiv -> « [TITRE] … » et « [RÉSUMÉ] … » ; rien n'est retiré."""
    entree = ET.fromstring(entree_atom)
    titre = _RE_ESPACES.sub(" ", entree.findtext(f"{_ATOM}title") or "").strip()
    resume = _RE_ESPACES.sub(" ", entree.findtext(f"{_ATOM}summary") or "").strip()
    return f"[TITRE] {titre}\n\n[RÉSUMÉ] {resume}"


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def nettoyer_source(brut: str, format_source: FormatSource) -> str:
    match format_source:
        case FormatSource.PMC_XML:
            return nettoyer_pmc_xml(brut)
        case FormatSource.PUBMED_XML:
            return nettoyer_pubmed_xml(brut)
        case FormatSource.ARXIV_LATEX:
            return nettoyer_latex(brut)
        case FormatSource.ARXIV_ABSTRACT:
            return nettoyer_resume_arxiv(brut)


_RE_RESUME_LATEX = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)


def extraire_resume(document: Document) -> str | None:
    """Résumé du document, pour donner à l'auditeur la population et le plan d'étude
    quand il ne reçoit que des fenêtres d'un texte intégral. Ne modifie pas texte_nettoye."""
    match document.format:
        case FormatSource.PMC_XML:
            article = _unique(ET.fromstring(document.texte_brut), "article")
            blocs: list[str] = []
            for resume in (e for e in article.iter() if _local(e.tag) == "abstract"):
                for enfant in resume:
                    _parcourir(enfant, blocs)
            return "\n\n".join(blocs) or None
        case FormatSource.PUBMED_XML | FormatSource.ARXIV_ABSTRACT:
            return document.texte_nettoye
        case FormatSource.ARXIV_LATEX:
            trouve = _RE_RESUME_LATEX.search(document.texte_nettoye)
            return trouve.group(1).strip() if trouve else None


def construire_document(source_id: str, format_source: FormatSource, brut: str) -> Document:
    return Document(
        source_id=source_id,
        format=format_source,
        texte_brut=brut,
        texte_nettoye=nettoyer_source(brut, format_source),
        textclean_version=config.TEXTCLEAN_VERSION,
    )
