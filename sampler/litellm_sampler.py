import os
import time
from typing import Any, Dict, List, Any

import openai
from openai import OpenAI
import litellm

from ..types import MessageList, SamplerBase, SamplerResponse


# https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses
def get_openai_web_search_tool(search_context_size: str = "medium") -> Dict[str, Any]:
    return {"type": "web_search_preview_2025_03_11", "search_context_size": search_context_size}


class LiteLLMSampler(SamplerBase):
    """
    Sample from LiteLLM
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        tools: List[Dict[str, str]] | None = None,
        extra_kwargs: Dict[str, Any] = {},
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.tools = tools
        self.reasoning_model = reasoning_model
        # extra kwargs for litellm.completion
        self.extra_kwargs = extra_kwargs

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        new_image = {
            "type": "input_image",
            "image_url": f"data:image/{format};{encoding},{image}",
        }
        return new_image

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                if self.reasoning_model:
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        max_tokens=self.max_tokens,
                        tools=self.tools,
                        **self.extra_kwargs,
                    )
                else:
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=self.tools,
                        **self.extra_kwargs,
                    )

                content = response['choices'][0]['message']['content']
                if content is None:
                    raise ValueError("Litellm API returned empty response; retrying")

                metadata = {"usage": response['usage']}
                if response['choices'][0]['message'].get('thinking_blocks') is not None:
                    extra_convo = [self._pack_message(x['type'], x['thinking']) for x in response['choices'][0]['message']['thinking_blocks']]
                    metadata["extra_convo"] = extra_convo
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata=metadata,
                    actual_queried_message_list=message_list,
                )
            except litellm.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
