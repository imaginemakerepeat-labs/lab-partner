# web_search.py

from __future__ import annotations

import re
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", "").strip(),
                    "url": (r.get("href") or r.get("url") or "").strip(),
                    "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                }
            )

    return results


def format_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return "I could not find any results."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['title']}\n{r['snippet']}\n{r['url']}")
    return "\n\n".join(lines)


def fetch_page_text(url: str, timeout: int = 10, max_paragraphs: int = 20) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    paragraphs = []
    for p in soup.find_all("p"):
        txt = " ".join(p.get_text(" ", strip=True).split())
        if len(txt) >= 60:
            paragraphs.append(txt)
        if len(paragraphs) >= max_paragraphs:
            break

    text = "\n".join(paragraphs)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 30]


def summarize_text(text: str, max_sentences: int = 5) -> str:
    """
    Simple extractive summary:
    - prefers earlier informative sentences
    - avoids very short/noisy lines
    """
    if not text:
        return "I could not extract readable article text."

    sentences = split_sentences(text)
    if not sentences:
        return "I could not build a summary from the page."

    scored = []
    for i, s in enumerate(sentences[:40]):
        score = 0

        # earlier sentences matter more
        score += max(0, 40 - i)

        # medium-length informative sentences score better
        word_count = len(s.split())
        if 12 <= word_count <= 40:
            score += 12
        elif 8 <= word_count <= 55:
            score += 6

        # light keyword preference
        lowered = s.lower()
        for kw in ["is", "are", "has", "have", "will", "announced", "released", "according"]:
            if kw in lowered:
                score += 2

        scored.append((score, i, s))

    scored.sort(reverse=True)

    chosen = sorted(scored[:max_sentences], key=lambda x: x[1])
    summary = " ".join(s for _, _, s in chosen)
    return summary.strip()


def search_and_extract(query: str, max_results: int = 5) -> dict:
    results = search_web(query, max_results=max_results)

    if not results:
        return {
            "mode": "web_context",
            "query": query,
            "results": [],
            "page": None,
        }

    top = results[0]

    page_data = None
    try:
        text = fetch_page_text(top["url"])

        # trim to keep token size small
        text = text[:2000]

        page_data = {
            "title": top["title"],
            "url": top["url"],
            "text": text,
        }

    except Exception:
        page_data = None

    return {
        "mode": "web_context",
        "query": query,
        "results": results[:3],
        "page": page_data,
    }