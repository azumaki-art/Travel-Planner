"""Shared web-search backend built on `ddgs`.

`ddgs` is occasionally flaky: the first request from a fresh process, or a burst
of requests, can come back empty or raise a rate-limit error. Every caller in this
project therefore goes through `search_text`, which retries with a small backoff
before giving up.
"""
import time
from typing import Dict, List

# English-first results, as this robot targets an international audience.
DEFAULT_REGION = "us-en"
DEFAULT_MAX_RESULTS = 10
_MAX_ATTEMPTS = 4


def search_text(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = DEFAULT_REGION,
    attempts: int = _MAX_ATTEMPTS,
) -> List[Dict[str, str]]:
    """Run a web search and return normalised ``{title, href, body}`` results.

    Retries transient failures. Raises RuntimeError only when every attempt errored;
    a genuinely empty result set is returned as an empty list.
    """
    from ddgs import DDGS

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with DDGS() as ddgs:
                raw = ddgs.text(
                    query,
                    region=region,
                    safesearch="moderate",
                    max_results=max_results,
                )
            results = [
                {
                    "title": str(item.get("title", "")),
                    "href": str(item.get("href", "")),
                    "body": str(item.get("body", "")),
                }
                for item in (raw or [])
            ]
            if results:
                return results
            last_error = "the search backend returned no results"
        except Exception as exc:  # network problem or backend rate limit
            last_error = exc

        if attempt < attempts:
            time.sleep(1.5 * attempt)

    if isinstance(last_error, Exception):
        raise RuntimeError(
            f"Web search for \"{query}\" failed after {attempts} attempts: {last_error}"
        ) from last_error
    return []
