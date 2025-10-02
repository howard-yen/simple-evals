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


SLIM_SYSTEM_MESSAGE = """You are a helpful assistant that can search the web. You are encourage to use the search tool to best answer the user's question. Use the search tool to collect useful information.
When using the search tool, you should think carefully about the question. Decompose and rewrite the search query if necessary. After using the search tool, you should reason about the results and summarize the relevant information to answer the user's question. If the search results are not relevant, you are encouraged to refine your search query and search again. Continue to use the tools until you have collected all the information you need, this may take many iterations.
The search tool will return a list of urls and their descriptions, and you should visit the urls that are relevant to the task. Visiting a url will provide you with more information.
After you have collected all the information you need, you should complete the given task."""


SLIM_SUMMARIZED_SYSTEM_MESSAGE = """You are a helpful assistant that can search the web. You are encourage to use the search tool to best answer the user's question. Use the search tool to collect useful information.
When using the search tool, you should think carefully about the question. Decompose and rewrite the search query if necessary. After using the search tool, you should reason about the results and summarize the relevant information to answer the user's question. If the search results are not relevant, you are encouraged to refine your search query and search again. Continue to use the tools until you have collected all the information you need, this may take many iterations.
The search tool will return a list of urls and their descriptions, and you should visit the urls that are relevant to the task. Visiting a url will provide you with more information.
After you have collected all the information you need, you should complete the given task.
You are given a summary of work done so far, which contains relevant information to the task. You should use this summary to continue the completion of the task."""


SUMMARY_SYSTEM_MESSAGE = """You are a helpful assistant that can summarize the information in the messages. You should summarize key information in the messages. Key information may include search queries issues, urls visited, and relevant results, but you may include other useful information as well. The summary will be given back to the original assistant in place of the messages to continue the completion of the task, so make sure to include all key and relevant information."""


class SlimSampler(SamplerBase):
    def __init__(
        self, 
        model: str, 
        system_message: str | None = None,
        max_iterations: int=10,
        max_tokens: int=1024,
        temperature: float=1.0,
        topk: int=10,
        content_length: int=10000,
        scoring_func: str="rouge",
        chunking_func: str="newline",
        summary_interval: int=50,
        summary_mode: str="turn",
        use_summary_system_message: bool=False,
        extra_kwargs: Dict[str, Any]={},
    ):
        self.model = model
        if use_summary_system_message:
            self.system_message = SLIM_SUMMARIZED_SYSTEM_MESSAGE
        else:
            self.system_message = system_message
        assert self.system_message, "System message is required for SlimSampler"
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.summary_interval = summary_interval
        self.summary_mode = summary_mode
        assert self.summary_mode in ["turn", "token", "none"], "Summary mode must be either turn or token or none"
        self.all_summaries = []
        self.extra_kwargs = extra_kwargs
        self.web_search_tool = WebSearchTool()
        self.topk = topk
        self.content_length = content_length
        self.scoring_func = scoring_func
        self.chunking_func = chunking_func


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


    def _summarize(self, message_list: MessageList) -> str:
        """
        Given a list of messages, summarize the information in the messages.
        """
        # first, construct the prompt, but exclude the original system message
        prompt = ""
        for message in message_list:
            if message['role'] == "developer" or message['role'] == "system":
                continue
            if message.get('tool_calls') is not None:
                func = message['tool_calls'][0]['function']
                if not isinstance(func, dict):
                    func = func.to_dict()
                prompt += f"<role>{message['role']}</role>\n<message>{message['content']}</message>\n<tool_calls>{json.dumps(func)}</tool_calls>\n\n"
            else:
                prompt += f"<role>{message['role']}</role>\n<message>{message['content']}</message>\n\n"
                
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_MESSAGE},
            {"role": "user", "content": f"Here is the conversation you should summarize. Do not complete the task, just summarize the information in the conversation:\n\n{prompt}"},
        ]
        response = self.generate(messages)
        return response

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        cur_iter = 0
        extra_convo = []
        all_usages = []
        summ_usages = []
        agent_usages = []
        tool_counts = defaultdict(int)
        fallback = False
        message_list = [
            self._pack_message("developer", self.system_message)
        ] + message_list
        original_message_list = copy.deepcopy(message_list)
        
        while cur_iter <= self.max_iterations:
            print(f"Iteration {cur_iter}\n")

            if cur_iter == self.max_iterations:
                response = self.generate(message_list)
            else:
                response = self.generate(message_list, tools=[SEARCH_TOOL, VISIT_TOOL])
            
            if isinstance(response, str):
                print(f"Error in iteration {cur_iter}. Falling back to not using tools.")
                response = self.generate(original_message_list)
                fallback = True
                if isinstance(response, str):
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": None, "fallback": True, "error": response},
                        actual_queried_message_list=original_message_list,
                    )

            message = response.choices[0].message
            tool_calls = message.get("tool_calls", None)
            all_usages.append(get_usage_dict(response.usage))
            agent_usages.append(get_usage_dict(response.usage))

            if message.get('reasoning_content'):
                reasoning_content = message.get('reasoning_content')
                extra_convo.append(self._pack_message("assistant thinking", reasoning_content))

            if tool_calls:
                message_list.append(message)
                for tool_call in tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"Function args: {function_args}")
                    tool_counts[tool_call.function.name] += 1

                    if tool_call.function.name == "search":
                        if "query" not in function_args:
                            tool_response = f"Error: Please provide a query to search for in the function arguments."
                        else:
                            tool_response = self.web_search_tool.search(function_args["query"])
                        
                    elif tool_call.function.name == "visit":
                        if "url" not in function_args:
                            tool_response = f"Error: Please provide a url to visit in the function arguments."
                        else:
                            tool_response = self.web_search_tool.open_url(
                                function_args["url"], function_args.get("query", ""), 
                                scoring_func=self.scoring_func, chunking_func=self.chunking_func
                            )
                    
                    else:
                        tool_response = f"Error: Unknown tool: {tool_call.function.name}. Only search and visit are allowed."

                    tool_message = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": tool_response,
                    }
                    
                    message_list.append(tool_message)
                    extra_convo.append(self._pack_message(f"tool_call {tool_call.function.name} {cur_iter}", tool_call.function.arguments))
                    extra_convo.append(self._pack_message("tool", tool_message['content']))

            else:
                print("No tools used")
                break

            # summarization step
            if (self.summary_mode == "turn" and (cur_iter+1) % self.summary_interval == 0) \
                or (self.summary_mode == "token" and response.usage.get("total_tokens", 0) > self.summary_interval):
                response  = self._summarize(message_list)
                if isinstance(response, str):
                    print("Error in summarization. Falling back to not summarizing.")
                else:
                    summ_usages.append(get_usage_dict(response.usage))
                    all_usages.append(summ_usages[-1])

                    self.all_summaries.append(response.choices[0].message.content)
                    summary_text = "Summary of the work done so far:\n\n" +  "\n".join([
                        f'Step {i+1}: {summary}' for i, summary in enumerate(self.all_summaries)
                    ])
                    message_list = copy.deepcopy(original_message_list)
                    message_list[0]['content'] = SLIM_SUMMARIZED_SYSTEM_MESSAGE
                    message_list.append(self._pack_message("user", summary_text))
                    extra_convo.append(self._pack_message("user", summary_text))

            cur_iter += 1

        metadata = {
            "fallback": fallback,
            "extra_convo": extra_convo,
            "usage": all_usages,
            "agent_usages": agent_usages,
            "summ_usages": summ_usages,
            "tool_counts": dict(tool_counts),
        }
        message = response['choices'][0]['message']
        response_text = message['content'] if message['content'] is not None else ""
        return SamplerResponse(
            response_text=response_text,
            response_metadata=metadata,
            actual_queried_message_list=original_message_list,
        )
        