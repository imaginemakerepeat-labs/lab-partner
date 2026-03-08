# web_search.py

from ddgs import DDGS


def search_web(query: str, max_results: int = 3):
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", "") or r.get("url", ""),
                "snippet": r.get("body", "") or r.get("snippet", ""),
            })

    return results


def format_results(results):
    if not results:
        return "I could not find any results."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"{i}. {r['title']}\n{r['snippet']}\n{r['url']}"
        )

    return "\n\n".join(lines)