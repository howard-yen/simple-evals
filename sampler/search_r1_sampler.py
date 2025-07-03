import time
import re
import requests
from typing import Any, Dict, List
import litellm
from transformers import AutoTokenizer

from ..types import MessageList, SamplerBase, SamplerResponse

# from: https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py
SEARCH_R1_SYSTEM_PROMPT = """You are a helpful assistant."""

SEARCH_R1_USER_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


class SearchR1Sampler(SamplerBase):
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
        search_endpoint: str = "http://127.0.0.1:8080/retrieve",
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
        self.tokenizer = AutoTokenizer.from_pretrained(model.replace("openai/", ""))
        
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
        self.search_stop_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    
    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}
    
    def _get_search_query(self, text: str) -> str | None:
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None
    
    def _search(self, query: str) -> str:
        try:
            payload = {
                "queries": [query],
                "topk": self.topk,
                "return_scores": True
            }
            results = requests.post(self.search_endpoint, json=payload).json()['result']
            
            def _passages2string(retrieval_result):
                format_reference = ''
                for idx, doc_item in enumerate(retrieval_result):
                    content = doc_item['document']['contents']
                    title = content.split("\n")[0]
                    text = "\n".join(content.split("\n")[1:])
                    format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
                return format_reference
            
            return _passages2string(results[0])
        except Exception as e:
            print(f"Search error: {e}")
            return ""
    
    def _generate_with_stop(self, prompt: str) -> str:
        trial = 0
        while True:
            try:
                self.extra_kwargs.update({
                    "stop": self.search_stop_sequences,
                })
                
                response = litellm.text_completion(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=7200,
                    **self.extra_kwargs
                )
                
                content = response['choices'][0]['text']
                if content is None:
                    raise ValueError("Litellm API returned empty response; retrying")

                if response['choices'][0]['stop_reason']:
                    content = content + response['choices'][0]['stop_reason']
                return content
                
            except Exception as e:
                exception_backoff = 2**trial
                exception_backoff = min(exception_backoff, 128)
                print(f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec: {e}")
                time.sleep(exception_backoff)
                trial += 1

                
    def flatten_message_list(self, message_list: MessageList) -> str:
        return "\n\n".join([f"{msg['role']}\n{msg['content']}" for msg in message_list])

    
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

        assert message_list[-1]['role'] == 'user', "Last message must be a user message"

        message_list[-1]['content'] = SEARCH_R1_USER_PROMPT + message_list[-1]['content']

        if self.tokenizer.chat_template:
            current_prompt = self.tokenizer.apply_chat_template(message_list, add_generation_prompt=True, tokenize=False, enable_thinking=self.reasoning_model)
        else:
            current_prompt = self.flatten_message_list(message_list)
            
        original_message_list = message_list.copy()

        iteration_count = 0
        extra_convo = []
        
        while iteration_count < self.max_iterations:
            # Generate response
            output_text = self._generate_with_stop(current_prompt)
            
            # Check if this contains a search query
            search_query = self._get_search_query(output_text)
            
            if search_query:
                # Perform search
                search_results = self._search(search_query)
                search_text = self.search_template.format(
                    output_text=output_text,
                    search_results=search_results
                )
                # note that in the search r1 framework, only the search text is added, thinking tokens are discarded
                current_prompt += search_text
                
                # Add to extra conversation for metadata
                extra_convo.append(self._pack_message(f"search_query_{iteration_count}", search_query))
                extra_convo.append(self._pack_message(f"search_results_{iteration_count}", search_results))
                
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
            "usage": None  # litellm usage tracking could be added here
        }
        
        return SamplerResponse(
            response_text=final_response,
            response_metadata=metadata,
            actual_queried_message_list=original_message_list,
        )
