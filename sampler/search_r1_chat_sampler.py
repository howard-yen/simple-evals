import time
import re
import requests
from typing import Any, Dict, List
import litellm

from ..types import MessageList, SamplerBase, SamplerResponse
from ..common import get_usage_dict
from ..tools.search_utils import WebSearchTool

# from: https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py
SEARCH_R1_SYSTEM_PROMPT = """You are a helpful assistant."""

SEARCH_R1_USER_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """

SEARCH_R1_USER_PROMPT_O3 = """Answer the given question. \
After reasoning, if you find you lack some knowledge, you can call a search engine by generating <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """
# You must conduct reasoning first every time you get new information. \

class SearchR1ChatSampler(SamplerBase):
    """
    SearchR1 sampler that implements reasoning and search loop using litellm
    """

    def __init__(
        self,
        model: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        system_message: str | None = SEARCH_R1_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_iterations: int = 100,
        search_endpoint: str = "http://127.0.0.1:8001/retrieve",
        reasoning_model: bool = False,
        topk: int = 3,
        extra_kwargs: Dict[str, Any] = {},
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.search_endpoint = search_endpoint
        self.topk = topk
        self.reasoning_model = reasoning_model
        self.extra_kwargs = extra_kwargs
        self.search_tool = WebSearchTool(topk=topk)

        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
        self.search_stop_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _get_search_query(self, text: str) -> str | None:
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None

    def _get_thinking(self, text: str) -> str | None:
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()

        pattern = re.compile(r"<think>(.*?)<search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None

    def _generate_with_stop(self, message_list: MessageList) -> str:
        trial = 0
        while True:
            try:
                self.extra_kwargs.update({
                    "stop": self.search_stop_sequences,
                })

                if self.reasoning_model:
                    self.extra_kwargs.pop("stop")
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        max_tokens=self.max_tokens,
                        timeout=3600,
                        **self.extra_kwargs
                    )
                else:
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=3600,
                        **self.extra_kwargs
                    )

                content = response['choices'][0]['message']['content']
                if content is None:
                    raise ValueError("Litellm API returned empty response; retrying")

                return content, get_usage_dict(response.usage), response._response_ms*1000

            except Exception as e:
                exception_backoff = 2**trial
                exception_backoff = min(exception_backoff, 128)
                print(f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec: {e}")
                time.sleep(exception_backoff)
                trial += 1


    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list

        if "o3" in self.model or "o4" in self.model:
            message_list[-1]['content'] = SEARCH_R1_USER_PROMPT_O3 + message_list[-1]['content']
        else:
            message_list[-1]['content'] = SEARCH_R1_USER_PROMPT + message_list[-1]['content']

        generation_time = 0
        tool_time = 0

        original_message_list = message_list.copy()
        current_message_list = message_list.copy()

        iteration_count = 0
        extra_convo = []
        # for multi-round frameworks, we keep track of all usages
        all_usage = []

        while iteration_count < self.max_iterations:
            # Generate response
            output_text, usage, generation_time = self._generate_with_stop(current_message_list)
            all_usage.append(usage)
            generation_time += generation_time

            # Check if this contains a search query
            thinking = self._get_thinking(output_text)
            if thinking:
                extra_convo.append(self._pack_message("assistant thinking", thinking))

            search_query = self._get_search_query(output_text)
            if search_query:
                start_time = time.time()
                # Perform search
                search_results = self.search_tool.search_r1(search_query, topk=self.topk)
                search_text = self.search_template.format(
                    output_text=output_text,
                    search_results=search_results
                )
                # note that in the search r1 framework, only the search text is added, thinking tokens are discarded
                current_message_list.append(self._pack_message("user", search_text))

                # Add to extra conversation for metadata
                extra_convo.append(self._pack_message(f"tool_call search", search_query))
                extra_convo.append(self._pack_message(f"tool", search_results))
                tool_time += time.time() - start_time
                iteration_count += 1
            else:
                # No search needed, we're done
                final_response = output_text
                break
        else:
            # Max iterations reached
            final_response = output_text

        metadata = {
            "iterations": iteration_count,
            "extra_convo": extra_convo,
            "all_usage": all_usage,
            "generation_time": generation_time,
            "tool_time": tool_time,
            "latency": generation_time + tool_time,
        }

        return SamplerResponse(
            response_text=final_response,
            response_metadata=metadata,
            actual_queried_message_list=original_message_list,
        )
