import os
import time
import json
from typing import Any, Dict, List, Any, Optional, Union
import copy
import random
import requests

from ..types import MessageList, SamplerBase, SamplerResponse
from ..common import get_usage_dict
from ..tools.search_utils import WebSearchTool

import litellm

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information. This tool will return a list of urls with their content.",
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

REACT_SYSTEM_MESSAGE = """You are a helpful assistant that can search the web. You are encouraged to use the search tool to best answer the user's question. Use the search tool to collect useful information.
When using the search tool, you should think carefully about the question. Decompose and rewrite the search query if necessary. After using the search tool, you should reason about the results and summarize the relevant information to answer the user's question. If the search results are not relevant, you should refine your search query and search again. Continue to use the tools until you have collected all the information you need, this may take many iterations.
The search tool will return a list of urls and their content. After you have collected all the information you need, you should complete the given task."""


class ReactSampler(SamplerBase):
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
                return f"Bad request error: {e}. Returning empty response."
            
            except litellm.APIConnectionError as e:
                print(f"API connection error: {e}. Returning empty response.")
                return f"API connection error: {e}. Returning empty response."

            except Exception as e:
                if trial >= 5:
                    return f"Error: {e}. Returning empty response after 5 trials."
                    
                exception_backoff = 2**trial  # exponential back off
                exception_backoff = min(exception_backoff, 120)
                print(f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec: {e}")
                time.sleep(exception_backoff)
                trial += 1


    def __call__(self, message_list: MessageList) -> SamplerResponse:
        cur_iter = 0
        extra_convo = []
        all_usages = []
        generation_time = 0
        tool_time = 0
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
        original_message_list = copy.deepcopy(message_list)
        
        while cur_iter <= self.max_iterations:
            print(f"Iteration {cur_iter}\n")
            if cur_iter == self.max_iterations:
                response = self.generate(message_list)
            else:
                response = self.generate(message_list, tools=[SEARCH_TOOL])

            import pdb; pdb.set_trace()
            if isinstance(response, str):
                print(f"Error in iteration {cur_iter}. Falling back to not using tools.")
                response = self.generate(original_message_list)
                tool_time = 0
                if isinstance(response, str):
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": None, "fallback": True, "error": response},
                        actual_queried_message_list=original_message_list,
                    )
                generation_time = response._response_ms*1000
            
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

                    if tool_call.function.name != "search":
                        tool_response = f"Error: {tool_call.function.name} is not a valid tool. Please use the search tool."
                    else:
                        if "query" not in function_args:
                            tool_response = f"Error: Please provide a query to search for in the function arguments."
                        else:
                            tool_response = self.web_search_tool.search_open_url(function_args["query"])

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

            cur_iter += 1

        metadata = {
            "fallback": False,
            "extra_convo": extra_convo,
            "usage": all_usages,
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
        