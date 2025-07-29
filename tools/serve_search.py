import os 
import json
import requests
import argparse
from functools import lru_cache
import time
import asyncio

import uvicorn 
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from urllib.parse import urlparse
from rouge_score import rouge_scorer
from diskcache import Cache

from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, BrowserConfig

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
cache = Cache()

class SearchRequest(BaseModel):
    query: str

class OpenUrlRequest(BaseModel):
    url: str
    query: str


def detect_content_type(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.path.lower().endswith('.pdf'):
        return 'pdf'

    try:
        response = requests.head(url, headers=HEADERS, timeout=(3, 5))
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        return "pdf" if "pdf" in content_type else "html"
    except Exception as e:
        return "html"


def find_snippet(texts: List[str], snippet: str, num_characters: int = 4000):
    # we iterate through the texts, calculate the ROUGE score between the snippet and the text
    # we mainly care about the recall score of ROUGE-L (most of the snippet is present in the long text)
    # take the text with the highest recall score and the surrounding text of the snippet
    positions = []
    start = 0
    best_recall = 0
    best_idx = 0
    for i, text in enumerate(texts):
        score = scorer.score(target=snippet, prediction=text)
        recall = score['rougeL'].recall
        if recall > best_recall:
            best_recall = recall
            best_idx = i
        positions.append((start, start + len(text)))
        start += len(text) + 1
    
    best_len = len(texts[best_idx])
    num_characters = num_characters - best_len
    final_text = []
    for i, pos in enumerate(positions):
        if (pos[0] >= positions[best_idx][0] - num_characters/2 and pos[1] <= positions[best_idx][1] + num_characters/2) or i == best_idx:
            final_text.append(texts[i])
    
    return "\n".join(final_text)
    

async def scrape_pdf(url: str, snippet: str | None = None, num_characters: int = 10000) -> Tuple[bool, str, str]:
    import fitz
    response = requests.get(url, headers=HEADERS, timeout=(3, 5))  # (connect timeout, read timeout)
    response.raise_for_status()
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    
    texts = text.split("\n")

    if snippet is not None:
        final_snippet = find_snippet(texts, snippet, num_characters)
    else:
        final_snippet = text

    return True, final_snippet, text


async def scrape_html(url: str, snippet: str | None = None, num_characters: int = 10000) -> Tuple[bool, str, str]:
    prune_filter = PruningContentFilter(threshold=0.4, threshold_type="dynamic", min_word_threshold=3)
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter, options={"ignore_links": False})
    browser_config = BrowserConfig(
        headless=True, verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--disable-extensions"]
    )
    crawler_config = CrawlerRunConfig(markdown_generator=md_generator, page_timeout=15000, verbose=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await asyncio.wait_for(crawler.arun(url=url, config=crawler_config), timeout=30)

    if not result.success:
        return False, f"Failed to scrape the page due to {result.error_message}", ""

    if len(result.markdown.raw_markdown.strip()) == 0:
        return False, f"Failed to scrape the page, returned empty content.", ""

    fit_markdown = result.markdown.fit_markdown 
    raw_markdown = result.markdown.raw_markdown

    if len(fit_markdown) > num_characters and snippet is not None:
        fit_markdown = find_snippet(fit_markdown.split("\n"), snippet, num_characters)

    return True, fit_markdown, raw_markdown

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 0.2

@lru_cache(maxsize=2048)
@cache.memoize(typed=True, expire=1e6, tag="search")
def _cached_search(query: str) -> str:
    """Cached search function that takes hashable parameters."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        'Content-Type': 'application/json'
    }

    payload = json.dumps({"q": query, "num": 10})

    for attempt in range(MAX_RETRIES):
        try: 
            response = requests.post(url=url, headers=headers, data=payload)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
                continue
            else:
                return f"Failed to search the query {query}.\nError: {str(e)}"
        except Exception as e:
            return f"Failed to search the query {query}.\nError: {str(e)}"
        
        if response.status_code in [408, 500, 502, 503, 504]:
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
                continue
            else:
                return f"Failed to search the query {query}.\nError: {response.text}"

    keys = ["title", "link", "snippet"]
    template = "<Search Result {position}>\n<Title: {title}>\n<URL: {link}>\n{snippet}\n</Search Result {position}>"

    response = response.json()
    results = response["organic"]
    results = [r for r in results if all(k in r for k in keys) and all(not isinstance(r[k], str) or len(r[k]) > 0 for k in keys)][:10]
    results = [{**r, "position": i+1} for i, r in enumerate(results)]
    output = "\n\n".join([template.format(**r) for r in results])
    output = f"The search engine returned {len(results)} results:\n\n" + output
    return output


@lru_cache(maxsize=2048)
@cache.memoize(typed=True, expire=1e6, tag="content")
def _cached_get_content(url: str) -> Tuple[bool, str, str]:
    """Cached function to get raw content from URL."""
    try:
        content_type = detect_content_type(url)
        if content_type == "pdf":
            result = asyncio.run(scrape_pdf(url, None, 10000))
        else:
            result = asyncio.run(scrape_html(url, None, 10000))
        return result
    except Exception as e:
        return False, str(e), ""


@lru_cache(maxsize=2048)
@cache.memoize(typed=True, expire=1e6, tag="snippet")
def _cached_find_snippet(content: str, query: str, num_characters: int = 10000) -> str:
    """Cached function to find snippet in content."""
    if not query:
        return content[:num_characters]
    
    content_lines = content.split("\n")
    return find_snippet(content_lines, query, num_characters)


def _cached_open_url(url: str, query: str) -> str:
    """Main function that combines cached content retrieval and snippet finding."""
    # First get the raw content (cached by URL only)
    success, content_or_error, raw_content = _cached_get_content(url)
    if not success:
        return f"Failed to open the url {url}.\nAdditional information: {content_or_error}"
    
    final_content = _cached_find_snippet(content_or_error, query, 10000)
    
    return f"Successfully opened the url {url}.\n<Content>\n{final_content}\n</Content>"


app = FastAPI()

@app.post("/search")
def search(request: SearchRequest):
    print(f"Search query: {request.query}")
    output = _cached_search(request.query)
    return {"output": output}


@app.post("/open_url")
def open_url(request: OpenUrlRequest): 
    print(f"Open url: {request.url} with query: {request.query}")
    output = _cached_open_url(request.url, request.query)
    return {'output': output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
