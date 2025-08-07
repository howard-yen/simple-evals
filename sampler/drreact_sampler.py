import os
import time
import json
from collections import defaultdict
from typing import Any, Dict, List, Any, Optional, Union
import copy
import random
import requests
import asyncio

from ..types import MessageList, SamplerBase, SamplerResponse
from ..common import get_usage_dict
from ..tools.search_utils import WebSearchTool, SEARCH_TOOL, VISIT_TOOL

import litellm


DRREACT_SYSTEM_MESSAGE = """You are a helpful assistant that can search the web. You are encourage to use the search tool to best answer the user's question. Use the search tool to collect useful information.
When using the search tool, you should think carefully about the question. Decompose and rewrite the search query if necessary. After using the search tool, you should reason about the results and summarize the relevant information to answer the user's question. If the search results are not relevant, you are encouraged to refine your search query and search again. Continue to use the tools until you have collected all the information you need, this may take many iterations.
The search tool will return a list of urls and their descriptions, and you should visit the urls that are relevant to the task. Visiting a url will provide you with more information.
After you have collected all the information you need, you should complete the given task."""


class DrReactSampler(SamplerBase):
    def __init__(
        self, 
        model: str, 
        system_message: str | None = None,
        max_iterations: int=10,
        max_tokens: int=1024,
        temperature: float=1.0,
        topk: int=10,
        track_queries: bool=False,
        extra_kwargs: Dict[str, Any]={},
    ):
        self.model = model
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.track_queries = track_queries
        self.extra_kwargs = extra_kwargs
        self.web_search_tool = WebSearchTool(topk=topk)


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
                    timeout=7200,
                    **kwargs
                )
                message = response['choices'][0]['message']
                if message['content'] is None and message.get("tool_calls") is None and message.get("reasoning_content") is None:
                    print(f"LiteLLM returned empty response: {response}")
                    raise ValueError("Litellm API returned empty response; retrying")
                
                return response

            except litellm.BadRequestError as e:
                print(f"Bad request error: {e}. Returning empty response.")
                return None
            
            except litellm.APIConnectionError as e:
                print(f"API connection error: {e}. Returning empty response.")
                return None

            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                exception_backoff = min(exception_backoff, 120)
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1


    def __call__(self, message_list: MessageList) -> SamplerResponse:
        cur_iter = 0
        extra_convo = []
        all_usages = []
        tool_counts = defaultdict(lambda: 0)
        generation_time = 0
        tool_time = 0
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
        original_message_list = copy.deepcopy(message_list)
        queries = set()
        
        while cur_iter < self.max_iterations:
            cur_iter += 1
            print(f"Iteration {cur_iter}\n")
            fallback = False
            if cur_iter == self.max_iterations:
                message_list.append(self._pack_message("user", "You have reached the maximum number of tool calls. Please complete the task without using any tools."))
                response = self.generate(message_list)
            else:
                response = self.generate(message_list, tools=[SEARCH_TOOL, VISIT_TOOL])
            
            if response is None:
                print(f"Error in iteration {cur_iter}. Falling back to not using tools.")
                response = self.generate(original_message_list)
                fallback = True
                generation_time = response._response_ms*1000
                tool_time = 0
                if response is None:
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": None, "fallback": True},
                        actual_queried_message_list=original_message_list,
                    )

            message = response.choices[0].message
            tool_calls = message.get("tool_calls", None)
            all_usages.append(get_usage_dict(response.usage))
            generation_time += response._response_ms*1000

            if message.get('reasoning_content'):
                reasoning_content = message.get('reasoning_content')
                extra_convo.append(self._pack_message("assistant thinking", reasoning_content))

            if tool_calls:
                message_list.append(message)
                start_time = time.time()
                for tool_call in tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"Function args: {function_args}")
                    tool_counts[tool_call.function.name] += 1

                    if tool_call.function.name == "search":
                        if "query" not in function_args:
                            tool_response = f"Error: Please provide a query to search for in the function arguments."
                        elif self.track_queries and function_args["query"] in queries:
                            tool_response = f"Error: You have already searched for this query. Please refine your search query."
                        else:
                            tool_response = self.web_search_tool.search(function_args["query"])
                            queries.add(function_args["query"])
                        
                    elif tool_call.function.name == "visit":
                        if "url" not in function_args:
                            tool_response = f"Error: Please provide a url to visit in the function arguments."
                        else:
                            tool_response = self.web_search_tool.open_url(function_args["url"], function_args.get("query", ""))
                    
                    else:
                        tool_response = f"Error: Unknown tool: {tool_call.function.name}"

                    tool_message = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": tool_response,
                    }
                    
                    message_list.append(tool_message)
                    extra_convo.append(self._pack_message(f"tool_call {tool_call.function.name} {cur_iter}", tool_call.function.arguments))
                    extra_convo.append(self._pack_message("tool", tool_message['content']))
                tool_time += time.time() - start_time

            else:
                print("No tools used")
                break

        metadata = {
            "fallback": fallback,
            "extra_convo": extra_convo,
            "all_usage": all_usages,
            "tool_counts": tool_counts,
            "tool_time": tool_time,
            "generation_time": generation_time,
            "latency": generation_time + tool_time,
        }
        message = response['choices'][0]['message']
        response_text = message['content'] if message['content'] is not None else ""
        return SamplerResponse(
            response_text=response_text,
            response_metadata=metadata,
            actual_queried_message_list=original_message_list,
        )
        