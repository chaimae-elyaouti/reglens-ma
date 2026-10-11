"""Chunking. Le cœur de la différenciation du projet.

Stratégie principale : UN CHUNK PAR ARTICLE.
Pourquoi : un article de loi est l'unité naturelle de citation. Découper au hasard (CharacterTextSplitter
du tuto) coupe les phrases et sépare une règle de son exception. Citer "Article 39" est aussi la seule
façon pour un lecteur de vérifier la source.

Stratégie de référence (baseline) : fenêtres de taille fixe, pour PROUVER par l'évaluation que
le chunking par article est meilleur (ou non !).
"""

from __future__ import annotations

import bisect
import hashlib
import re

from reglens.schemas import Chunk

# "Article 39", "Article premier", y compris quand l'extraction PDF colle le texte ("Article 4Les ...").
# Sensible à la casse : "l'article 9" (renvoi dans une phrase) n'est PAS un début d'article.
ARTICLE_RE = re.compile(r"(?<![\w'’«\"(])Article\s+(premier|\d{1,3})(?!\d)")

# "LIVRE II", "TITRE premier", "CHAPITRE 3", "Section 2"... en début de ligne.
HEADING_RE = re.compile(
    r"(?m)^[ \t]*((?i:livre|titre|chapitre|section|sous[- ]section)\s+"
    r"(?:(?i:premier|première)|[IVXLC]+|\d+)\b[^\n]{0,120})$"
)
_LEVELS = {"livre": 0, "titre": 1, "chapitre": 2, "section": 3, "sous": 4}
_LABEL_FIX = re.compile(r"^(Article\s+(?:premier|\d{1,3}))(?=[A-ZÉÈÀÂÎÔÙ])")
_SENT_SPLIT = re.compile(r"(?<=[\.;:!?])\s+(?=[A-ZÉÈÀÂÎÔÙ0-9\-–•])|\n+")


def _num(label: str) -> int:
    return 1 if label.lower() == "premier" else int(label)


def find_article_starts(
    text: str, first_article: int = 1, max_gap: int = 10
) -> list[tuple[int, int]]:
    """Positions (offset, numéro) des vrais débuts d'article.

    Problème : le mot "Article N" apparaît aussi dans des renvois ("Article 12 ci-dessus..."). Les articles
    d'un code sont numérotés dans l'ordre : on n'accepte donc un candidat que s'il suit le dernier numéro
    accepté. Un saut (article manquant dans le PDF) est toléré seulement si le numéro intermédiaire
    n'apparaît nulle part plus loin dans le texte.
    """
    cands = [(m.start(), _num(m.group(1))) for m in ARTICLE_RE.finditer(text)]
    accepted: list[tuple[int, int]] = []
    last = first_article - 1
    for i, (pos, n) in enumerate(cands):
        if n <= last:
            continue
        if n == last + 1:
            ok = True
        elif n <= last + max_gap:
            ok = not any(last < n2 < n for _, n2 in cands[i + 1 :])
        else:
            ok = False
        if ok:
            accepted.append((pos, n))
            last = n
    return accepted


def _level(heading: str) -> int:
    return _LEVELS.get(heading.split()[0].lower().split("-")[0], 3)


def _section_at(headings: list[tuple[int, str]], pos: int) -> str:
    state: dict[int, str] = {}
    for hpos, htxt in headings:
        if hpos >= pos:
            break
        lvl = _level(htxt)
        state[lvl] = htxt
        for other in [k for k in state if k > lvl]:
            del state[other]
    return " > ".join(state[k] for k in sorted(state))


def split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Découpe un article trop long aux frontières de phrases/alinéas, avec léger chevauchement."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    if overlap >= max_chars:
        raise ValueError("overlap doit être < max_chars")
    pieces = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        while len(p) > max_chars:  # phrase géante : coupe dure
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:max_chars])
            p = p[max_chars - overlap :]
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            if " " in tail:
                tail = tail[tail.find(" ") + 1 :]  # ne coupe pas un mot en deux
            cur = f"{tail} {p}".strip() if len(tail) + 1 + len(p) <= max_chars else p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def _page_at(offsets: list[int] | None, pos: int) -> int | None:
    if not offsets:
        return None
    return max(1, bisect.bisect_right(offsets, pos))


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def build_chunks_by_article(
    text: str,
    *,
    doc_id: str,
    doc_title: str,
    source_url: str = "",
    lang: str = "fr",
    as_of: str = "",
    page_offsets: list[int] | None = None,
    max_chars: int = 1500,
    overlap: int = 150,
    with_header: bool = True,
    first_article: int = 1,
) -> list[Chunk]:
    """Un chunk par article. `with_header` ajoute "Titre du texte — Article N (section)" au texte embeddé :
    le vecteur "sait" ainsi de quel texte/article/chapitre il vient (et BM25 peut trouver "article 39")."""
    starts = find_article_starts(text, first_article)
    headings = [(m.start(), re.sub(r"\s+", " ", m.group(1)).strip()) for m in HEADING_RE.finditer(text)]
    boundaries = sorted({p for p, _ in starts} | {h for h, _ in headings} | {len(text)})

    chunks: list[Chunk] = []
    for pos, n in starts:
        end = boundaries[bisect.bisect_right(boundaries, pos)]
        body = _LABEL_FIX.sub(r"\1 ", text[pos:end].strip(), count=1)
        section = _section_at(headings, pos)
        for j, part in enumerate(split_long(body, max_chars, overlap), 1):
            header = f"{doc_title} — Article {n}"
            if section:
                header += f" ({section})"
            if j > 1:
                header += " [suite]"
            embed_text = f"{header}\n{part}" if with_header else part
            chunks.append(
                Chunk(
                    id=f"{doc_id}:a{n}" + (f"-p{j}" if j > 1 else ""),
                    doc_id=doc_id,
                    doc_title=doc_title,
                    article=str(n),
                    part=j,
                    section_path=section,
                    text=part,
                    embed_text=embed_text,
                    source_url=source_url,
                    lang=lang,
                    as_of=as_of,
                    page=_page_at(page_offsets, pos),
                    embed_hash=_sha1(embed_text),
                )
            )
    return chunks


def build_chunks_fixed(
    text: str,
    *,
    doc_id: str,
    doc_title: str,
    source_url: str = "",
    lang: str = "fr",
    as_of: str = "",
    page_offsets: list[int] | None = None,
    size: int = 1000,
    overlap: int = 100,
    first_article: int = 1,
) -> list[Chunk]:
    """Baseline : fenêtres de taille fixe qui ignorent les articles (sans en-tête de contexte).

    On étiquette quand même chaque fenêtre avec l'article où elle commence ET tous les articles qu'elle
    recouvre (`covers`), pour pouvoir la comparer à la même vérité terrain que le chunking par article."""
    starts = find_article_starts(text, first_article)
    start_pos = [p for p, _ in starts]
    chunks: list[Chunk] = []
    step = size - overlap
    for idx, off in enumerate(range(0, len(text), step)):
        window = text[off : off + size].strip()
        if not window:
            continue
        end = min(off + size, len(text))
        k = bisect.bisect_right(start_pos, off) - 1
        first = starts[k][1] if k >= 0 else 0
        covered = {first} | {n for p, n in starts if off <= p < end}
        chunks.append(
            Chunk(
                id=f"{doc_id}:w{idx:04d}",
                doc_id=doc_id,
                doc_title=doc_title,
                article=str(first),
                covers=[str(a) for a in sorted(covered)],
                text=window,
                embed_text=window,
                source_url=source_url,
                lang=lang,
                as_of=as_of,
                page=_page_at(page_offsets, off),
                embed_hash=_sha1(window),
            )
        )
    return chunks
