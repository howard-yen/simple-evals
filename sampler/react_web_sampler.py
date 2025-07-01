import os
import time
import json
from typing import Any, Dict, List, Any, Optional, Union
import copy
import random
import requests
import asyncio

from retrievaltools.utils import scrape_page_content_crawl4ai
from retrievaltools import load_retriever, RetrieverOptions

from ..types import MessageList, SamplerBase, SamplerResponse

import litellm

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information. This tool will return a list of urls that are relevant to the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user question or search query."
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
        "name": "open_url",
        "description": "Open a url and optionally search for a specific query. By default, this tool will return the beginning of the page, but searching for a specific query will return the relevant part of the page that contains the query text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url to open."
                },
                "query": {
                    "type": "string",
                    "description": "The query to search for on the opened url. This should be the exact text that you are looking for."
                }
            },
            "required": [
                "url",
            ],
            "additionalProperties": False
        },
    }
}   

REACT_WEB_SYSTEM_MESSAGE = """You are a helpful assistant that can search the web. You are encourage to use the search tool to best answer the user's question. Use the search tool to collect useful information.
When using the search tool, you should think carefully about the question, decompose and rewrite the search query if necessary. After using the search tool, you should reason about the results and summarize the relevant information to answer the user's question. If the search results are not relevant, you are encouraged to refine your search query and search again.
The search tool will return a list of urls, and you should visit the urls that are relevant to the task. To get more information from the url, you should use the visit tool.
After you have collected all the information you need, you may first summarize and reason about the information, and then write your final answer."""


class ReactWebSampler(SamplerBase):
    def __init__(
        self, 
        model: str, 
        system_message: str | None = None,
        max_iterations: int=10,
        max_tokens: int=1024,
        temperature: float=1.0,
        topk: int=10,
        extra_kwargs: Dict[str, Any]={},
    ):
        self.model = model
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs

        self.retriever_options = RetrieverOptions(
            retriever_type="web",
            include_corpus=False,
            port=8000,
            topk=topk,
            use_cache=False,
            web_scraping="none",
            verbose=False,
        )
        self.retriever = load_retriever(self.retriever_options)


    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def generate(self, message_list: MessageList, **kwargs):
        trial = 0
        while True:
            try:
                kwargs.update(self.extra_kwargs)
                response = litellm.completion(
                    model=self.model,
                    messages=message_list,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **kwargs
                )
                message = response['choices'][0]['message']
                if message['content'] is None and message.get("tool_calls") is None:
                    raise ValueError("Litellm API returned empty response; retrying")
                return response

            except litellm.BadRequestError as e:
                print("Bad Request Error", e)
                raise e
                
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        cur_iter = 0
        extra_convo = []
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
        original_message_list = copy.deepcopy(message_list)
        
        while cur_iter < self.max_iterations:
            print(f"Iteration {cur_iter}\n")
            fallback = False
            try:
                response = self.generate(message_list, tools=[SEARCH_TOOL, VISIT_TOOL])
            except Exception as e:
                print(f"Error in iteration {cur_iter}: {e}. Falling back to not using tools.")
                # it's possible that this generate call will fail, we need to handle this case.
                try:
                    response = self.generate(original_message_list)
                    fallback = True
                    break
                except Exception as fallback_error:
                    print(f"Fallback response also failed: {fallback_error}. Returning empty response.")
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": None, "fallback": True},
                        actual_queried_message_list=original_message_list,
                    )

            cur_iter += 1
            # if search tool, call retriever, otherwise return the response
            message = response.choices[0].message
            tool_calls = message.get("tool_calls")

            # if we reached the max iterations and we still call the tool, we fallback to not using tools
            if tool_calls and cur_iter == self.max_iterations:
                print("Fallback to not using tools")
                response = self.generate(original_message_list)
                fallback = True
                
            elif tool_calls:
                print("Using tools")
                message_list.append(message)
                for tool_call in tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"Function args: {function_args}")

                    if tool_call.function.name == "search":
                        function_response = self.retriever.retrieve(**function_args)[0]
                        tool_message = {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": self.retriever.format_results(function_response, topk=self.retriever_options.topk),
                            }
                        
                    elif tool_call.function.name == "open_url":
                        if "url" not in function_args:
                            tool_response = f"Error: Please provide a url to open in the function arguments."
                        else:
                            function_response = asyncio.run(scrape_page_content_crawl4ai(function_args["url"], function_args.get("query", ""), verbose=False))
                            success, snippet, fulltext = function_response
                            if success:
                                tool_response = f"Successfully opened the url {function_args['url']}.<Content>\n{snippet}\n</Content>"
                            else:
                                tool_response = f"Failed to open the url {function_args['url']}."

                        tool_message = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": tool_response,
                        }
                        
                    message_list.append(tool_message)
                    extra_convo.append(self._pack_message(f"tool call iter {cur_iter} {tool_call.function.name}", tool_call.function.arguments))
                    extra_convo.append(self._pack_message("tool", tool_message['content']))

            else:
                print("No tools used")
                break

        metadata = {
            "fallback": fallback,
            "extra_convo": extra_convo,
            "usage": response.usage
        }
        message = response['choices'][0]['message']
        return SamplerResponse(
            response_text=message['content'],
            response_metadata=metadata,
            actual_queried_message_list=original_message_list,
        )
        