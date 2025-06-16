import os
import time
import json
from typing import Any, Dict, List, Any, Optional, Union
import copy
import random
import requests

from retrievaltools import load_retriever, RetrieverOptions

from ..types import MessageList, SamplerBase, SamplerResponse

import litellm

SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
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
}]


class ReactSampler(SamplerBase):
    def __init__(
        self, 
        model: str, 
        max_iterations: int=10,
        max_tokens: int=1024,
        temperature: float=1.0,
        topk: int=10,
        extra_kwargs: Dict[str, Any]={},
    ):
        self.model = model
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
            use_crawl4ai=True,
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
        original_message_list = copy.deepcopy(message_list)
        
        while cur_iter < self.max_iterations:
            print(f"Iteration {cur_iter}")
            fallback = False
            response = self.generate(message_list, tools=SEARCH_TOOL)
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
                    function_response = self.retriever.retrieve(**function_args)[0]
                    tool_message = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": self.retriever.format_results(function_response, topk=self.retriever_options.topk),
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
        