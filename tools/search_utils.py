import json
import time
import requests


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information. This tool will return a list of urls with a snippet of the content in the url.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
            },
            "required": [
                "query",
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

SEARCH_RESPONSE_TOOL = {
    "type": SEARCH_TOOL['type'],
    "name": SEARCH_TOOL['function']['name'],
    "description": SEARCH_TOOL['function']['description'],
    "parameters": SEARCH_TOOL['function']['parameters'],
    "strict": SEARCH_TOOL['function']['strict'],
}

VISIT_TOOL = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Visit a url and optionally search for a specific query. If given an empty query, this tool will return the beginning of the page, but searching for a specific query will return the relevant part of the page that contains the query text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url to open."
                },
                "query": {
                    "type": "string",
                    "description": "The query to search for in the url. The tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query."
                }
            },
            "required": [
                "url",
            ],
            "additionalProperties": False
        },
    }
}

VISIT_TOOL_NO_QUERY = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Visit a url and return the page content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url to open."
                }
            },
            "required": [
                "url",
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

VISIT_RESPONSE_TOOL = {
    "type": VISIT_TOOL['type'],
    "name": VISIT_TOOL['function']['name'],
    "description": VISIT_TOOL['function']['description'],
    "parameters": VISIT_TOOL['function']['parameters'],
}

VISIT_RESPONSE_TOOL_NO_QUERY = {
    "type": VISIT_TOOL_NO_QUERY['type'],
    "name": VISIT_TOOL_NO_QUERY['function']['name'],
    "description": VISIT_TOOL_NO_QUERY['function']['description'],
    "parameters": VISIT_TOOL_NO_QUERY['function']['parameters'],
}

def _ensure_list(value):
    """Return value as a list if it isn't one already."""
    if isinstance(value, list):
        return value
    return [value]


class WebSearchTool():
    def __init__(self, port: int=8006, max_retries: int=3, timeout: int=1500):
        self.url = f"http://localhost:{port}"
        self.max_retries = max_retries
        self.timeout = timeout

    def _post_with_retry(self, endpoint: str, payload: str) -> requests.Response:
        for attempt in range(self.max_retries):
            try:
                return requests.post(self.url + endpoint, data=payload, headers={"Content-Type": "application/json"}, timeout=self.timeout)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Request to {endpoint} failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def _search_single(self, query: str, topk: int = 10) -> str:
        if not query or not query.strip():
            return json.dumps({"error": "Please provide a query to search for."})

        payload = json.dumps({"query": query, "topk": topk})
        response = self._post_with_retry("/search", payload)
        return response.json()['output']

    def search(self, query, topk: int = 10):
        """Search the web for information. Accepts a single query string or a list of queries."""
        queries = _ensure_list(query)
        results = [self._search_single(q, topk) for q in queries]
        return results if len(results) > 1 else results[0]

    def _open_url_single(self, url: str, query: str = "", content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
        if not url or not isinstance(url, str) or not url.strip():
            return "Please provide a url to open."

        payload = {"url": url, "query": query, "content_length": content_length, "scoring_func": scoring_func, "chunking_func": chunking_func}
        payload = json.dumps(payload)
        response = self._post_with_retry("/open_url", payload)
        try:
            out = response.json()
            return out['output']
        except Exception as e:
            print("Open url error: " + str(e))
            print(response)
            print(response.text)
            return "Open url error: " + str(e)

    def open_url(self, url, query: str = "", content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline"):
        """Open a url and optionally search for a specific query. Accepts a single url string or a list of urls."""
        urls = _ensure_list(url)
        results = [self._open_url_single(u, query, content_length, scoring_func, chunking_func) for u in urls]
        return results if len(results) > 1 else results[0]

    def _search_open_url_single(self, query: str, topk: int = 10, content_length: int = 10000) -> str:
        if not query or not query.strip():
            return "Search error: Please provide a query to search for."

        payload = json.dumps({"query": query, "topk": topk, "content_length": content_length})
        response = self._post_with_retry("/search_open_url", payload)
        return response.json()['output']

    def search_open_url(self, query, topk: int = 10, content_length: int = 10000):
        """Search the web for information, and also open all the urls. Accepts a single query or a list of queries."""
        queries = _ensure_list(query)
        results = [self._search_open_url_single(q, topk, content_length) for q in queries]
        return results if len(results) > 1 else results[0]

    def _search_o1_single(self, query: str, topk: int = 10):
        if not query or not query.strip():
            return json.dumps({"output": "Search error: Please provide a query to search for.", "search_results": []})

        payload = json.dumps({"query": query, "topk": topk})
        response = self._post_with_retry("/search_o1", payload)
        try:
            out = response.json()
            return out
        except Exception as e:
            print("Search o1 error: " + str(e))
            print(response)
            print(response.text)
            return {"output": "Search error: " + str(e), "search_results": []}

    def search_o1(self, query, topk: int = 10):
        """Search the web for information. Accepts a single query or a list of queries."""
        queries = _ensure_list(query)
        results = [self._search_o1_single(q, topk) for q in queries]
        # flatten the search results
        res = [r for result in results for r in result['search_results']]
        return {"output": "\n\n".join([result['output'] for result in results]), "search_results": res}
        # return results if len(results) > 1 else results[0]
