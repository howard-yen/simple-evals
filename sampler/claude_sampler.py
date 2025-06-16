import time
import os

import anthropic

from typing import Dict, List, Any
from ..types import MessageList, SamplerBase, SamplerResponse
from .. import common


def get_anthropic_web_search_tool(max_uses: int = 5) -> List[Dict[str, Any]]:
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }


CLAUDE_SYSTEM_MESSAGE_LMSYS = (
    "The assistant is Claude, created by Anthropic. The current date is "
    "{currentDateTime}. Claude's knowledge base was last updated in "
    "March 2025 and it answers user questions about events before "
    "March 2025 and after March 2025 the same way a highly informed "
    "individual from March 2025 would if they were talking to someone "
    "from {currentDateTime}. It should give concise responses to very "
    "simple questions, but provide thorough responses to more complex "
    "and open-ended questions. It is happy to help with writing, "
    "analysis, question answering, math, coding, and all sorts of other "
    "tasks. It uses markdown for coding. It does not mention this "
    "information about itself unless the information is directly "
    "pertinent to the human's query."
).format(currentDateTime="2025-06-01")
# reference: https://github.com/lm-sys/FastChat/blob/7899355ebe32117fdae83985cf8ee476d2f4243f/fastchat/conversation.py#L894

                
def parse_content_block(content):
    """Parse different types of ContentBlock into string representation"""
    def format_input(input_obj, max_length=200):
        """Format input object with key-value pairs on separate lines"""
        if isinstance(input_obj, dict):
            items = []
            for key, value in input_obj.items():
                if isinstance(value, str) and len(value) > max_length:
                    # Truncate long strings for readability
                    value = value[:max_length] + "..."
                items.append(f"  {key}: {value}")
            return "{\n" + "\n".join(items) + "\n}"
        elif isinstance(input_obj, list):
            return "[\n" + "\n".join(input_obj) + "\n]"
        else:
            return str(input_obj)
    
    def format_content(content_obj, max_length=200):
        """Format content with proper truncation"""
        if isinstance(content_obj, list):
            return "[\n" + "\n".join(format_content(item) for item in content_obj) + "\n]"
        
        if "websearchresult" in type(content_obj).__name__.lower():
            return f"[Web Search Result]\nTitle: {content_obj.title}\nURL: {content_obj.url}"
        elif "codeexecutiontoolresult" in type(content_obj).__name__.lower():
            return f"[Code Execution Result]\nCode: {content_obj.code}\nOutput: {content_obj.output}"
        elif "mcp_tool_use" in type(content_obj).__name__.lower():
            formatted_input = format_input(content_obj.input)
            return f"[MCP Tool Use: {content_obj.name}]\nInput: {formatted_input}"
        elif "mcp_tool_result" in type(content_obj).__name__.lower():
            return f"[MCP Tool Result]\nContent: {format_content(content_obj.content)}"

        content_str = str(content_obj)
        if len(content_str) > max_length:
            return content_str[:max_length] + "..."
        return content_str
    
    if content.type == "text":
        return content.text
    elif content.type == "thinking":
        return content.thinking
    elif content.type == "redacted_thinking":
        return f"[Redacted Thinking]\nData: {content.data}"
    elif content.type == "tool_use":
        formatted_input = format_input(content.input)
        return f"[Tool Use: {content.name}]\nID: {content.id}\nInput: {formatted_input}"
    elif content.type == "server_tool_use":
        formatted_input = format_input(content.input)
        return f"[Server Tool Use: {content.name}]\nID: {content.id}\nInput: {formatted_input}"
    elif content.type == "web_search_tool_result":
        formatted_content = format_content(content.content)
        return f"[Web Search Results]\nTool Use ID: {content.tool_use_id}\nContent: {formatted_content}"
    elif content.type == "code_execution_tool_result":
        formatted_content = format_content(content.content)
        return f"[Code Execution Result]\nTool Use ID: {content.tool_use_id}\nContent: {formatted_content}"
    elif content.type == "container_upload":
        return f"[Container Upload]\nFile ID: {content.file_id}"
    elif content.type == "mcp_tool_use":
        formatted_input = format_input(content.input)
        return f"[MCP Tool Use: {content.name}]\nServer: {content.server_name}\nID: {content.id}\nInput: {formatted_input}"
    elif content.type == "mcp_tool_result":
        error_status = "[ERROR] " if content.is_error else "[SUCCESS] "
        formatted_content = format_content(content.content)
        return f"[MCP Tool Result] {error_status}\nTool Use ID: {content.tool_use_id}\nContent: {formatted_content}"
    else:
        return f"[Unknown Content Type: {content.type}]\n{str(content)}"


class ClaudeCompletionSampler(SamplerBase):

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        temperature: float = 0.0,  # default in Anthropic example
        max_tokens: int = 4096,
        thinking_budget: int | None = None,
        tools: List[Dict[str, str]] = [],
    ):
        self.client = anthropic.Anthropic(timeout=1800)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "base64"
        if thinking_budget:
            self.thinking = {
                "budget_tokens": thinking_budget,
                "type": "enabled",
            }
            # cannot change the temperature when thinking is enabled
            self.temperature = 1
        else:
            self.thinking = {"type": "disabled"}
        self.tools = tools

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image",
            "source": {
                "type": encoding,
                "media_type": f"image/{format}",
                "data": image,
            },
        }
        return new_image

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        
        while True:
            try:
                if not common.has_only_user_assistant_messages(message_list):
                    raise ValueError(f"Claude sampler only supports user and assistant messages, got {message_list}")
                if self.system_message:
                    response_message = self.client.messages.create(
                        model=self.model,
                        system=self.system_message,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=message_list,
                        thinking=self.thinking,
                        tools=self.tools,
                    )
                    claude_input_messages: MessageList = [{"role": "system", "content": self.system_message}] + message_list
                else:
                    response_message = self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=message_list,
                        thinking=self.thinking,
                        tools=self.tools,
                    )
                    claude_input_messages = message_list
                
                metadata = {}
                extra_convo = []
                
                # Group consecutive text blocks, process others
                processed_blocks = []
                i = 0
                while i < len(response_message.content):
                    block = response_message.content[i]
                    if block.type == "text":
                        # Collect consecutive text blocks
                        j = i
                        while j < len(response_message.content) and response_message.content[j].type == "text":
                            j += 1
                        text_parts = [response_message.content[k].text for k in range(i, j)]
                        processed_blocks.append(("text", "\n".join(text_parts)))
                        i = j
                    else:
                        processed_blocks.append((block.type, parse_content_block(block)))
                        i += 1
                
                # Last text block becomes response_text, others go to extra_convo
                if processed_blocks and processed_blocks[-1][0] == "text":
                    response_text = processed_blocks[-1][1]
                    extra_convo = [self._pack_message(f"assistant {t}", c) for t, c in processed_blocks[:-1]]
                else:
                    # if we don't end with a text block, we retry
                    continue
                
                if extra_convo:
                    metadata["extra_convo"] = extra_convo
                    
                return SamplerResponse(
                    response_text=response_text,
                    response_metadata=metadata,
                    actual_queried_message_list=claude_input_messages,
                )
            except anthropic.RateLimitError as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            except anthropic.InternalServerError as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Internal server error so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
