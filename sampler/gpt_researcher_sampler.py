import asyncio
import time
from typing import Any, Dict

from gpt_researcher import GPTResearcher

from ..types import MessageList, SamplerBase, SamplerResponse

GPT_RESEARCHER_SYSTEM_MESSAGE = """
You are a helpful assistant.
""".strip()

class CustomLogsHandler:
    """A custom Logs handler class to handle JSON data."""
    def __init__(self):
        self.logs = []  # Initialize logs to store data

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON data and log it."""
        self.logs.append(data)  # Append data to logs

class GPTResearcherSampler(SamplerBase):
    """
    Sample using GPT Researcher for web research
    """

    def __init__(
        self,
        report_type: str = "deep",
        config_path: str | None = None,
        system_message: str | None = None,
        verbose: bool = False,
    ):
        self.report_type = report_type
        self.config_path = config_path
        self.system_message = system_message
        self.verbose = verbose

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    async def _get_report(self, question: str) -> dict[str, Any]:
        """Get research report from GPT Researcher"""
        # Initialize researcher with specified research type
        logs_handler = CustomLogsHandler()
        researcher_kwargs = {
            "query": question,
            "report_type": self.report_type,
            "websocket": logs_handler,
            "verbose": self.verbose,
        }
        
        if self.config_path:
            researcher_kwargs["config_path"] = self.config_path
            
        # Each run requires its own instance of GPTResearcher
        researcher = GPTResearcher(**researcher_kwargs)
        
        # Run research
        research_data = await researcher.conduct_research()

        # Access research sources
        sources = researcher.get_research_sources()

        # Get visited URLs
        urls = researcher.get_source_urls()
        
        # Generate report
        report = await researcher.write_report()

        return {
            "report": report,
            "sources": sources,
            "urls": urls,
            "research_data": research_data,
            "logs": logs_handler.logs,
            "costs": researcher.get_costs(),
            "researcher": researcher,
        }

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
 
        # If there's only one message, use it directly
        if len(message_list) == 1:
            question = message_list[0].get("content", "")
        else:
            # Concatenate all messages with their roles
            question_parts = []
            for msg in message_list:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                question_parts.append(f"[{role}]: {content}")
            question = "\n\n".join(question_parts)
        
        trial = 0
        while True:
            try:
                # Run async research in sync context
                research_results = asyncio.run(self._get_report(question))
                logs = research_results["logs"]
                extra_convo = [{"role": f"{x['type']} {x['content']}", "content": x['output']} for x in logs if x['type'] == "logs"]
                costs = [x for x in logs if x['type'] == "cost"]

                return SamplerResponse(
                    response_text=research_results["report"],
                    response_metadata={
                        "sources": research_results["sources"],
                        "urls": research_results["urls"],
                        "research_data": research_results["research_data"],
                        "extra_convo": extra_convo,
                        "costs": costs,
                        "logs": logs,
                    },
                    actual_queried_message_list=message_list,
                )

            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Research exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                
                # Prevent infinite retries
                if trial > 5:
                    return SamplerResponse(
                        response_text=f"Research failed after {trial} attempts: {str(e)}",
                        response_metadata={
                            "error": str(e),
                            "sources": [],
                            "urls": [],
                            "research_data": None,
                            "costs": 0,
                        },
                        actual_queried_message_list=message_list,
                    )
