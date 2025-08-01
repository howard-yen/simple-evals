import json
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

class WebSearchTool():
    def __init__(self, topk: int=10, port: int=8006):
        self.topk = topk
        self.url = f"http://localhost:{port}"

    def search(self, query: str) -> str:
        """Search the web for information. This tool will return a list of urls that are relevant to the query."""
        if not query or not query.strip():
            return json.dumps({"error": "Please provide a query to search for."})

        payload = json.dumps({"query": query})
        response = requests.post(self.url + "/search", data=payload)
        return response.json()['output']

    def open_url(self, url: str, query: str = "") -> str:
        """Open a url and optionally search for a specific query. By default, this tool will return the beginning of the page, but searching for a specific query will return the relevant part of the page that contains the query text."""
        if not url or not url.strip():
            return "Please provide a url to open."

        payload = {"url": url, "query": query}
        payload = json.dumps(payload)
        response = requests.post(self.url + "/open_url", data=payload)
        return response.json()['output']
