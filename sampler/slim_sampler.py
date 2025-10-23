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

from slim import Slim
import litellm


class SlimSampler(SamplerBase):
    def __init__(
        self, 
        model: str, 
        system_message: str | None = None,
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
        extra_kwargs: Dict[str, Any]={},
    ):
        self.agent = Slim(
            model=model,
            system_message=system_message,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            temperature=temperature,
            topk=topk,
            content_length=content_length,
            scoring_func=scoring_func,
            chunking_func=chunking_func,
            summary_interval=summary_interval,
            summary_mode=summary_mode,
            use_summary_system_message=use_summary_system_message,
            extra_kwargs=extra_kwargs,
        )


    def __call__(self, message_list: MessageList) -> SamplerResponse:
        return self.agent(message_list)
       
    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}
