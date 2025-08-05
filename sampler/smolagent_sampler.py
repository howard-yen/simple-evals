import json
import os
import time
from typing import Any, Dict, Optional

from huggingface_hub import login

from .smolagents_scripts.text_inspector_tool import TextInspectorTool
from .smolagents_scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from .smolagents_scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
)
from smolagents.memory import MemoryStep

from ..types import MessageList, SamplerBase, SamplerResponse
from ..common import get_usage_dict

SMOLAGENT_CODEAGENT_SYSTEM_MESSAGE = """
You are a helpful assistant.
For your final response, remember to use the `final_answer` tool to provide your final answer. The tool can be called with the following format:
```python
final_answer("Your final response here")
```
""".strip()

SMOLAGENT_JSONAGENT_SYSTEM_MESSAGE = """
You are a helpful assistant.
For your final response, remember to use the `final_answer` tool to provide your final answer. The tool can be called with the following format:
{
  "name": "final_answer",
  "arguments": {"answer": "insert your final answer here"}
}
""".strip()

class SmolAgentSampler(SamplerBase):
    """
    Sample from SmolAgents with configurable parameters and proper error handling
    This sampler uses the HuggingFace Open Deep Research agent
    https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research
    """

    def __init__(
        self,
        model: str = "o1",
        system_message: str | None = None,
        config_path: str | None = None,
        max_completion_tokens: int = 32768,
        max_steps: int = 20,
        verbosity_level: int = 1,
        planning_interval: int = 4,
        text_limit: int = 100000,
        viewport_size: int = 5120,
        timeout: int = 300,
        downloads_folder: str = "downloads_folder",
    ):
        # Model configuration
        self.model = model
        self.system_message = system_message
        self.max_completion_tokens = max_completion_tokens
        
        # Agent configuration
        self.max_steps = max_steps
        self.verbosity_level = verbosity_level
        self.planning_interval = planning_interval
        self.text_limit = text_limit
        
        # Browser configuration
        self.viewport_size = viewport_size
        self.timeout = timeout
        self.downloads_folder = downloads_folder
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Initialize HuggingFace login
        # login(os.environ.get("HF_TOKEN"))
        
        # Setup browser configuration
        self._setup_browser_config()
        
    def _load_config(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update parameters from config
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    def _setup_browser_config(self) -> None:
        """Setup browser configuration"""
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        
        self.browser_config = {
            "viewport_size": self.viewport_size,
            "downloads_folder": self.downloads_folder,
            "request_kwargs": {
                "headers": {"User-Agent": user_agent},
                "timeout": self.timeout,
            },
        }
        
        # Create downloads folder
        os.makedirs(f"./{self.downloads_folder}", exist_ok=True)

    def _create_agent(self):
        """Create and configure the SmolAgent"""
        custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
        
        model_params = {
            "model_id": self.model,
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": self.max_completion_tokens,
        }
        model = LiteLLMModel(**model_params)

        browser_metadata = {"all_usage": [], "tool_calls": []}
        manager_metadata = {"all_usage": [], "tool_calls": []}
        def tracking_callback(memory_step, agent, metadata):
            usage = get_usage_dict(memory_step.token_usage)
            usage['step_type'] = str(type(memory_step))
            metadata['all_usage'].append(usage)
            if hasattr(memory_step, 'tool_calls'):
                metadata['tool_calls'].extend([x.dict() for x in memory_step.tool_calls])

        # Setup browser and web tools
        browser = SimpleTextBrowser(**self.browser_config)
        web_tools = [
            GoogleSearchTool(provider="serper"),
            VisitTool(browser),
            PageUpTool(browser),
            PageDownTool(browser),
            FinderTool(browser),
            FindNextTool(browser),
            ArchiveSearchTool(browser),
            TextInspectorTool(model, self.text_limit),
        ]
        
        # Create web browser agent
        text_webbrowser_agent = ToolCallingAgent(
            model=model,
            tools=web_tools,
            max_steps=self.max_steps,
            verbosity_level=self.verbosity_level,
            planning_interval=self.planning_interval,
            name="search_agent",
            description="""A team member that will search the internet to answer your question.
        Ask him for all your questions that require browsing the web.
        Provide him as much context as possible, in particular if you need to search on a specific timeframe!
        And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
        Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
        """,
            provide_run_summary=True,
            step_callbacks={MemoryStep: [lambda memory_step, agent: tracking_callback(memory_step, agent, browser_metadata)]},
        )
        
        # Add custom prompt template
        text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

        # Create manager agent
        manager_agent = CodeAgent(
            model=model,
            tools=[visualizer, TextInspectorTool(model, self.text_limit)],
            max_steps=12,
            verbosity_level=self.verbosity_level,
            additional_authorized_imports=["*"],
            planning_interval=self.planning_interval,
            managed_agents=[text_webbrowser_agent],
            return_full_result=True,
            step_callbacks={MemoryStep: [lambda memory_step, agent: tracking_callback(memory_step, agent, manager_metadata)]},
        )

        return manager_agent, {"browser_metadata": browser_metadata, "manager_metadata": manager_metadata}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        """Pack message in the expected format"""
        return {"role": role, "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """Execute the agent with proper error handling and retry logic"""
        # The current agent does not support parallel calls as the memory and steps are shared.
        # For each call, we need to create a new agent.
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        
        trial = 0
        max_retries = 3
        
        # If there's only one message, use it directly
        if len(message_list) == 1:
            user_query = message_list[0].get("content", "")
        else:
            # Concatenate all messages with their roles
            messages = []
            for msg in message_list:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                messages.append(f"[{role}]: {content}")
            user_query = "\n\n".join(messages)

        while trial < max_retries:
            agent, metadata = self._create_agent()
            try:
                # Run the agent
                result = agent.run(user_query)
                response_text = result.output
                if response_text is None:
                    raise Exception("No response text, retrying...")

                # HF SmolAgent may return int or other types
                response_text = str(response_text)

                # in HF SmolAgent, the messages are the steps. Each step contains one conversation.
                # We will save the very last step as the actual queried message list, but we can also save all the steps.
                messages = result.messages
                history = []

                # the first message is just the task, which appears again in the second message
                for message in messages[1:]:
                    history.append({
                        "input": [
                            {"role": x['role'], "content": "\n".join([y["text"] for y in x['content']])} if isinstance(x, dict) else {"role": x.role, "content": "\n".join([y["text"] for y in x.content])}
                            for x in message['model_input_messages']
                        ],
                        "output": {"role": message['model_output_message']['role'], "content": message['model_output_message']['content']}
                    })
                extra_convo = history[-1]['input']
               
                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={
                        "extra_convo": extra_convo,
                        "memory": history,
                        "latency": result.timing.duration,
                        **metadata,
                    },
                    actual_queried_message_list=message_list,
                )
                
            except Exception as e:
                raise e
                trial += 1
                exception_backoff = 2 ** trial
                print(
                    f"SmolAgent exception on trial {trial}/{max_retries}, "
                    f"waiting {exception_backoff} seconds: {e}"
                )
                
                if trial >= max_retries:
                    return SamplerResponse(
                        response_text="",
                        response_metadata={
                            "error": str(e),
                            "max_retries_exceeded": True,
                            "trial": trial,
                        },
                        actual_queried_message_list=message_list,
                    )
                
                time.sleep(exception_backoff)
        
        # Should not reach here, but just in case
        return SamplerResponse(
            response_text="",
            response_metadata={"error": "Unknown error"},
            actual_queried_message_list=message_list,
        )