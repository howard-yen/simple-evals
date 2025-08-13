import time
import re
import json
from typing import Any, Dict, List
import litellm

from ..types import MessageList, SamplerBase, SamplerResponse
from ..common import get_usage_dict
from ..tools.search_utils import WebSearchTool

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information. The system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.",
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

class SearchO1ToolChatSampler(SamplerBase):
    """
    Search O1 sampler that implements reasoning and search loop using litellm
    """

    def __init__(
        self,
        model: str = "o1",
        system_message: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_search_limit: int = 10,
        reasoning_model: bool = False,
        topk: int = 10,
        extra_kwargs: Dict[str, Any] = {},
    ):
        self.model = model
        if system_message is None:
            system_message = self._get_multiqa_search_o1_instruction(max_search_limit)
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_search_limit = max_search_limit
        self.topk = topk
        self.reasoning_model = reasoning_model
        self.extra_kwargs = extra_kwargs
        self.search_tool = WebSearchTool(topk=topk)

        self.search_template = '\n\n{output_text}<|begin_search_result|>{search_results}<|end_search_result|>\n\n'
        self.search_stop_sequences = ["<|end_search_query|>", " <|end_search_query|>", "<|end_search_query|>\n", " <|end_search_query|>\n"]

    def _get_multiqa_search_o1_instruction(self, max_search_limit: int) -> str:
        return (
            "You are a reasoning assistant with the ability to perform web searches to help "
            "you answer the user's question accurately. You have special tools:\n\n"
            "- To perform a search, use the search tool.\n"
            "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
            f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {max_search_limit}.\n\n"
            "Once you have all the information you need, continue your reasoning.\n\n"
            "Example:\n"
            "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
            "Assistant thinking steps:\n"
            "- I need to find out who voices Lara Croft in the video game.\n"
            "- Then, I need to determine which company developed that video game.\n\n"
            "Assistant:\n"
            "search(query='Alice David Lara Croft voice')\n\n"
            "(System returns processed information from relevant web pages)\n\n"
            "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
            "Assistant:\n"
            "search(query='video game developed by Alice David Lara Croft')\n\n"
            "(System returns processed information from relevant web pages)\n\n"
            "Assistant continues reasoning with the new information...\n\n"
            "Remember:\n"
            "- Use the search tool to request a web search.\n"
            "- When done searching, continue your reasoning.\n\n"
        )

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _get_webpage_to_reasonchain_instruction(self, prev_reasoning: str, search_query: str, document: str) -> str:
        return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

    def _extract_answer(self, text: str, mode: str = 'infogen') -> str:
        """Extract information from model output based on mode"""
        extracted_text = ''
        if mode == 'codegen':
            # Extract the code between ```python and ```
            pattern = r'```python\s*(.*?)\s*```'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                extracted_text = matches[-1].strip()  # Take the last match
        elif mode == 'infogen':
            # Extract content after **Final Information** or **Modified Reasoning Steps**
            pattern_info = "**Final Information**"
            pattern_step = "**Modified Reasoning Steps**"
            if pattern_info in text:
                extracted_text = text.split(pattern_info)[-1].replace("\n","").strip("```").strip()
            elif pattern_step in text:
                extracted_text = text.split(pattern_step)[-1].strip("```").strip()
            else:
                extracted_text = "No helpful information found."
        else:
            # Existing extraction logic for 'gen' and 'choose' modes
            pattern = r'\\boxed\{(.*)\}'
            matches = re.findall(pattern, text)
            if matches:
                extracted_text = matches[-1]  # Take the last match
                if mode in ['choose', 'qa']:
                    # Handle 'choose' mode
                    inner_pattern = r'\\text\{(.*)\}'
                    inner_matches = re.findall(inner_pattern, extracted_text)
                    if inner_matches:
                        extracted_text = inner_matches[-1]  # Take the last match
                    extracted_text = extracted_text.strip("()")
        return extracted_text

       
    def _generate_with_stop(self, message_list: MessageList, tools = None) -> str:
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
                        timeout=600,
                        tools=tools,
                        **self.extra_kwargs
                    )
                else:
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=600,
                        tools=tools,
                        **self.extra_kwargs
                    )

                message = response['choices'][0]['message']
                if message['content'] is None and message.get("tool_calls") is None and message.get("reasoning_content") is None:
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


    def _generate_webpage_analysis(self, prev_reasoning: str, search_query: str, document: str, extra_convo: list[dict[str, Any]]) -> str:
        """Generate webpage analysis using litellm"""
        instruction = self._get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document)
        extra_convo.append(self._pack_message("user", instruction))
        message = self._pack_message("user", instruction)
        return self._generate_with_stop([message])        


    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

        generation_time = 0
        tool_time = 0

        original_message_list = message_list.copy()
        all_output_text = ''

        search_count = 0
        executed_search_queries = set()
        extra_convo = []
        # for multi-round frameworks, we keep track of all usages
        all_usage = []

        while True:
            # Generate response
            response = self._generate_with_stop(message_list, tools=[SEARCH_TOOL])
            if isinstance(response, str):
                print(f"Error in generation: {response}")
                return SamplerResponse(
                    response_text="",
                    response_metadata={"usage": None, "error": response},
                    actual_queried_message_list=message_list,
                )
            usage = get_usage_dict(response.usage)
            response_time = response._response_ms*1000

            message = response['choices'][0]['message']
            output_text = message['content']
            all_usage.append(usage)
            generation_time += response_time
            message_list.append(message)

            # Check if this contains a search query
            if message.get('reasoning_content'):
                extra_convo.append(self._pack_message("assistant thinking", message['reasoning_content']))
                all_output_text += message['reasoning_content']

            if output_text is not None:
                all_output_text += output_text
            tool_calls = message.get('tool_calls', None)
            if tool_calls:
                for tool_call in tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    extra_convo.append(self._pack_message(f"tool_call {tool_call.function.name}", function_args))

                    if tool_call.function.name != "search":
                        tool_response = f"Error: Unknown tool: {tool_call.function.name}"
                        message_list.append({'tool_call_id': tool_call.id, 'role': 'tool', 'name': tool_call.function.name, 'content': tool_response})
                        extra_convo.append(self._pack_message("tool", tool_response))
                        continue

                    if "query" not in function_args:
                        tool_response = f"Error: Please provide a query to search for in the function arguments."
                        message_list.append({'tool_call_id': tool_call.id, 'role': 'tool', 'name': tool_call.function.name, 'content': tool_response})
                        extra_convo.append(self._pack_message("tool", tool_response))
                        continue

                    search_query = function_args['query']
                    if search_count < self.max_search_limit and search_query not in executed_search_queries:
                        start_time = time.time()
                        # Perform search
                        formatted_documents = self.search_tool.search_o1(search_query, topk=self.topk)
                        tool_time += time.time() - start_time

                        # when reasoning about the results, we use the previous reasoning steps
                        all_reasoning_steps = all_output_text.replace('\n\n', '\n').split("\n")
                        truncated_prev_reasoning = ""
                        for i, step in enumerate(all_reasoning_steps):
                            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"
                        
                        prev_steps = truncated_prev_reasoning.split('\n\n')
                        if len(prev_steps) <= 5:
                            truncated_prev_reasoning = '\n\n'.join(prev_steps)
                        else:
                            truncated_prev_reasoning = ''
                            for i, step in enumerate(prev_steps):
                                if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                    truncated_prev_reasoning += step + '\n\n'
                                else:
                                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                        truncated_prev_reasoning += '...\n\n'
                            truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')
                        
                        # the search results are the formatted documents, which we perform reasoning on
                        reasoning_response, reasoning_usage, reasoning_time = self._generate_webpage_analysis(truncated_prev_reasoning, search_query, formatted_documents, extra_convo)
                        if reasoning_response is None:
                            print("Bad Request Error in reasoning, returning empty response")
                            return SamplerResponse(
                                response_text="",
                                response_metadata={"usage": None, "error": "Bad Request Error"},
                                actual_queried_message_list=message_list,
                            )
                            
                        reasoning_output = reasoning_response['choices'][0]['message']['content']
                        extracted_info = f"{BEGIN_SEARCH_RESULT}{self._extract_answer(reasoning_output, mode='infogen')}{END_SEARCH_RESULT}"
                        all_usage.append(reasoning_usage)
                        generation_time += reasoning_time

                        # Add search result to conversation
                        message_list.append({'tool_call_id': tool_call.id, 'role': 'tool', 'name': tool_call.function.name, 'content': extracted_info})

                        # Add to extra conversation for metadata
                        extra_convo.append(self._pack_message(f"tool", extracted_info))
                        search_count += 1
                        executed_search_queries.add(search_query)
                    
                    elif search_count >= self.max_search_limit:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        extra_convo.append(self._pack_message("tool", limit_message))
                        message_list.append({'tool_call_id': tool_call.id, 'role': 'tool', 'name': tool_call.function.name, 'content': limit_message})
                        
                    elif search_query in executed_search_queries:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        extra_convo.append(self._pack_message("tool", limit_message))
                        message_list.append({'tool_call_id': tool_call.id, 'role': 'tool', 'name': tool_call.function.name, 'content': limit_message})

            else:
                # No search needed, we're done
                final_response = output_text
                break
                
        metadata = {
            "iterations": search_count,
            "extra_convo": extra_convo,
            "usage": all_usage,
            "generation_time": generation_time,
            "tool_time": tool_time,
            "latency": generation_time + tool_time,
        }

        return SamplerResponse(
            response_text=final_response,
            response_metadata=metadata,
            actual_queried_message_list=original_message_list,
        )
