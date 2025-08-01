import os
import time
import warnings
from typing import Any, Dict, List, Any

import openai
from openai import OpenAI
import litellm

from ..types import MessageList, SamplerBase, SamplerResponse
from ..common import get_usage_dict

# Suppress repeated Google Cloud SDK warnings
warnings.filterwarnings("once", category=UserWarning, module="google.auth._default")


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
        # https://github.com/BerriAI/litellm/blob/ef7f8cce9340a596d4fdae3496c6c84dcc4100c4/litellm/llms/base_llm/base_utils.py#L175
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                start_time = time.time()
                if self.reasoning_model:
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        max_tokens=self.max_tokens,
                        tools=self.tools,
                        timeout=7200,
                        **self.extra_kwargs,
                    )
                else:
                    response = litellm.completion(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=self.tools,
                        timeout=7200,
                        **self.extra_kwargs,
                    )

                metadata = {"usage": get_usage_dict(response['usage']), "latency": time.time() - start_time}
                message = response['choices'][0]['message']
                content = message['content']

                if content is None:
                    if message.get('reasoning_content') is not None:
                        content = ""
                        extra_convo = [self._pack_message('assistant thinking', message['reasoning_content'])]
                        metadata["extra_convo"] = extra_convo
                    else:
                        raise ValueError("Litellm API returned empty response; retrying")
                
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
                exception_backoff = min(exception_backoff, 128)
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
