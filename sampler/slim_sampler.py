import os
import time
import json
from collections import defaultdict
from typing import Any, Dict, List, Any, Optional, Union
import copy
import random
import requests
import asyncio
import openai

from ..types import MessageList, SamplerBase, SamplerResponse
from ..common import get_usage_dict
from ..tools.search_utils import WebSearchTool, SEARCH_TOOL, VISIT_TOOL, SEARCH_RESPONSE_TOOL, VISIT_RESPONSE_TOOL, VISIT_TOOL_NO_QUERY, VISIT_RESPONSE_TOOL_NO_QUERY

# from slim import Slim
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
        summary_system_message: str | None = None,
        max_iterations: int=10,
        max_tokens: int=1024,
        temperature: float=1.0,
        summary_interval: int=50,
        summary_mode: str="turn",
        topk: int=10,
        content_length: int=10000,
        scoring_func: str="rouge",
        chunking_func: str="newline",
        use_summary_system_message: bool=False,
        use_responses_api: bool=False,
        keep_reasoning: bool=False,
        base_url: str | None = None,
        search_tool: dict | None = None,
        visit_tool: dict | None = None,
        no_visit_tool: bool=False,
        no_query_in_visit: bool=False,
        extra_kwargs: Dict[str, Any]={},
    ):
        self.model = model
        if use_summary_system_message:
            self.system_message = SLIM_SUMMARIZED_SYSTEM_MESSAGE
        else:
            self.system_message = system_message if system_message is not None else SLIM_SYSTEM_MESSAGE

        if summary_system_message is not None:
            self.summary_system_message = SLIM_SUMMARIZED_SYSTEM_MESSAGE
        else:
            self.summary_system_message = summary_system_message

        if search_tool is None:
            self.search_tool = SEARCH_TOOL if not use_responses_api else SEARCH_RESPONSE_TOOL
        else:
            self.search_tool = search_tool
        if visit_tool is None:
            if no_query_in_visit:
                self.visit_tool = VISIT_TOOL_NO_QUERY if not use_responses_api else VISIT_RESPONSE_TOOL_NO_QUERY
            else:
                self.visit_tool = VISIT_TOOL if not use_responses_api else VISIT_RESPONSE_TOOL
        else:
            self.visit_tool = visit_tool
        self.tools = [self.search_tool, self.visit_tool] if not self.no_visit_tool else [self.search_tool]

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
        self.use_responses_api = use_responses_api
        self.keep_reasoning = keep_reasoning
        self.base_url = base_url


    def __call__(self, message_list: MessageList) -> SamplerResponse:
        return self.agent(message_list)
       
    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}


    def generate(self, message_list: MessageList, **kwargs):
        trial = 0
        while True:
            try:
                kwargs.update(self.extra_kwargs)
                if self.use_responses_api:
                    client = openai.OpenAI(base_url=self.base_url)
                    response = client.responses.create(
                        model = self.model,
                        input = message_list,
                        max_output_tokens = self.max_tokens,
                        temperature = self.temperature,
                        **kwargs
                    )
                    if response.output[-1].type == 'reasoning':
                        import pdb; pdb.set_trace()
                        raise Exception("need to retry")

                else:
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        base_url=self.base_url,
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
            {"role": "system", "content": self.summary_system_message},
            {"role": "user", "content": f"{prompt}"},
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
                response = self.generate(message_list, tools=self.tools)
            
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

            all_usages.append(get_usage_dict(response.usage))
            agent_usages.append(get_usage_dict(response.usage))
            if self.use_responses_api:
                tool_calls = []
                for item in response.output:
                    message_list.append(item)
                    if item.type == 'function_call':
                        tool_calls.append(item)
                    elif item.type == 'reasoning':
                        extra_convo.append(self._pack_message("assistant thinking", item.content[0]['text']))
                    else:
                        output_text = response.output_text
                
            else: 
                message = response.choices[0].message
                tool_calls = message.get("tool_calls", None)

                if message.get('reasoning_content'):
                    reasoning_content = message.get('reasoning_content')
                    extra_convo.append(self._pack_message("assistant thinking", reasoning_content))
                    if self.keep_reasoning:
                        message_list.append(self._pack_message("assistant", reasoning_content))
                
                if tool_calls:
                    message_list.append(message)

            if tool_calls:
                for tool_call in tool_calls:
                    tool_response = None
                    try:
                        if self.use_responses_api:
                            function_name = tool_call.name
                            function_args = json.loads(tool_call.arguments)
                        else:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                        print(f"Function args: {function_args}")
                    except json.decoder.JSONDecodeError:
                        if self.use_responses_api:
                            function_args = tool_call.arguments
                        else:
                            function_args = tool_call.function.arguments
                        tool_response = f"Error: Invalid JSON in tool call:\n{function_args}"

                    tool_counts[function_name] += 1

                    if tool_response is None:
                        if function_name == "search":
                            if "query" not in function_args:
                                tool_response = f"Error: Please provide a query to search for in the function arguments."
                            else:
                                if isinstance(function_args["query"], list):
                                    tool_response = "\n=======\n".join([self.web_search_tool.search(query, topk=self.topk) for query in function_args["query"]])
                                else:
                                    tool_response = self.web_search_tool.search(function_args["query"], topk=self.topk)
                            
                        elif function_name == "visit":
                            if "url" not in function_args:
                                tool_response = f"Error: Please provide a url to visit in the function arguments."
                            else:

                                if isinstance(function_args["url"], list):
                                    tool_response = "\n=======\n".join([self.web_search_tool.open_url(
                                        url, function_args.get("query", function_args.get("goal", "")), 
                                        scoring_func=self.scoring_func, chunking_func=self.chunking_func) 
                                        for url in function_args["url"]
                                    ])
                                else:
                                    tool_response = self.web_search_tool.open_url(
                                        function_args["url"], function_args.get("query", ""), 
                                        scoring_func=self.scoring_func, chunking_func=self.chunking_func
                                    )   
                        else:
                            tool_response = f"Error: Unknown tool: {function_name}. Only search and visit are allowed."

                    if self.use_responses_api:
                        tool_message = {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": json.dumps({"content": tool_response}),
                            "role": "tool",
                            "name": function_name,
                        }
                    else:
                        tool_message = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_response,
                        }
                    
                    message_list.append(tool_message)
                    extra_convo.append(self._pack_message(f"tool_call {function_name} {cur_iter}", function_args))
                    extra_convo.append(self._pack_message("tool", tool_response))

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

                    if self.use_responses_api:
                        self.all_summaries.append(response.output_text)
                    else:
                        self.all_summaries.append(response.choices[0].message.content)
                    summary_text = "Summary of the work done so far:\n\n" +  "\n".join([
                        f'Step {i+1}: {summary}' for i, summary in enumerate(self.all_summaries)
                    ])
                    message_list = copy.deepcopy(original_message_list)
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
        response_text = message.get('content', None)
        return SamplerResponse(
            response_text=response_text,
            response_metadata=metadata,
            actual_queried_message_list=original_message_list,
        )
        
