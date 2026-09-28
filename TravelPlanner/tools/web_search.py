from typing import Annotated

from langchain_core.tools import tool

from tools.search_backend import search_text


@tool
def web_search(
    keywords: Annotated[str, "keywords to search for, determined based on your current task objective, be as precise and detailed as possible"],
    max_results: Annotated[int, "maximum number of search results to return, between 1 and 20. If the returned search results do not have much useful information, you can specify to return more search results"] = 10,
) -> list:
    """Web search tool. Searches the web in English for the given keywords and
returns a list of search results, each containing the page title, link and an
opening snippet of the page content."""
    max_results = max(1, min(int(max_results), 20))
    results = search_text(keywords, max_results=max_results)
    if not results:
        return f"No web results found for \"{keywords}\". Try different or broader keywords."
    return results


# test the tool
if __name__ == "__main__":
    print(web_search.args_schema.model_json_schema())
    a = web_search.invoke({"keywords": "Tokyo attractions opening hours"})
    print(a)
