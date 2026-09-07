import json
import os
import urllib.parse

from env.base import Environment
from worker.tool import Tool


def make_web_tools(env: Environment) -> list[Tool]:
    def _fetch_url(url: str) -> str:
        return env.http_get(url)

    def _web_search(query: str, num_results: int = 5) -> list[dict]:
        """SerpAPI-backed search. Requires SERPAPI_API_KEY env var.
        Uses Environment.http_get so that env swaps still intercept network."""
        key = os.getenv("SERPAPI_API_KEY", "")
        if not key:
            return [{"error": "SERPAPI_API_KEY not set"}]
        q = urllib.parse.urlencode({"q": query, "api_key": key, "num": num_results})
        try:
            raw = env.http_get(f"https://serpapi.com/search.json?{q}")
            data = json.loads(raw)
            out = []
            for item in (data.get("organic_results") or [])[:num_results]:
                out.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                })
            return out
        except Exception as e:
            return [{"error": f"web_search failed: {type(e).__name__}: {e}"}]

    return [
        Tool(
            name="fetch_url",
            description="GET a URL and return its body as text.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "HTTP(S) URL to fetch."}},
                "required": ["url"],
                "additionalProperties": False,
            },
            func=_fetch_url,
        ),
        Tool(
            name="web_search",
            description="Search the web via SerpAPI and return [{title, link, snippet}].",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "num_results": {"type": "integer", "description": "Max results to return.", "default": 5, "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            func=_web_search,
        ),
    ]
