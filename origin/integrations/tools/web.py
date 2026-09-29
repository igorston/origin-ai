"""Internet tools: `web_search` and `fetch_url` (see origin.integrations.web).

They are "network" tools: the engine only offers them to the model when the request
asks for it (`use_web`, the "Acesso à internet" switch), so nothing leaves the machine
unless the user turned that on. WEB_ACCESS=false removes them altogether.
"""

from typing import TYPE_CHECKING

import httpx
from langchain_core.tools import BaseTool, tool

from origin.integrations.web import READ_FAILED, UNTRUSTED, WebClient, WebError

if TYPE_CHECKING:
    from origin.integrations.registry import ToolContext

NETWORK = {"network": True}


def get_tools(ctx: "ToolContext") -> list[BaseTool]:
    settings = ctx.settings
    if not settings.web_access:
        return []
    client = WebClient(settings)

    @tool
    async def web_search(query: str) -> str:
        """Search the internet. Returns titles, URLs and snippets of the top results.

        Use when the answer depends on current or changing information: exchange rates and
        prices ("cotação do dólar hoje"), weather ("vai chover amanhã?"), news, scores and
        results, latest versions and releases, schedules, anything "hoje", "atual", "último",
        "agora" that is not just the date or time. Also for facts you are not sure about, or
        when the user asks to search ("pesquise", "procure", "busque na internet").
        Do NOT use for general knowledge you are sure of, math, creative writing, or facts
        about the user (those are in memory).

        Args:
            query: a short search query, in the language most likely to find the answer.
        """
        try:
            results = await client.search(query, settings.web_max_results)
        except (WebError, httpx.HTTPError) as exc:
            return f"Search failed: {exc}"
        if not results:
            return f"No results for {query!r}."
        lines = [f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, 1)]
        return (
            f"{UNTRUSTED}\nResults for {query!r}:\n\n" + "\n".join(lines) + "\n\n"
            "Answer from these results and cite the URLs you use. If the snippets are not "
            "enough, read the most relevant page with fetch_url."
        )

    @tool
    async def fetch_url(url: str) -> str:
        """Read a web page and return its text (truncated).

        Use when the user gives a URL or asks about a specific page, or to read a search
        result in full when its snippet is not enough.

        Args:
            url: the full http(s) address of the page.
        """
        try:
            final, title, text = await client.fetch(url)
        except (WebError, httpx.HTTPError) as exc:
            return f"{READ_FAILED} {url}: {exc}"
        limit = settings.web_fetch_max_chars
        cut = f"\n[... truncated: {len(text) - limit} more characters]" if len(text) > limit else ""
        return f"{UNTRUSTED}\nURL: {final}\nTitle: {title or '-'}\n\n{text[:limit]}{cut}"

    # Outside content: after it, changes to the user's data need the user's own request, and
    # only addresses from the user or from these results can be opened (agent.TurnGuard).
    web_search.metadata = {**NETWORK, "untrusted": True}
    fetch_url.metadata = {**NETWORK, "untrusted": True, "url_arg": "url"}
    return [web_search, fetch_url]
