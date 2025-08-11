import asyncio
import time
import gc
from typing import Any, Dict

import openai
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

    async def _get_report_with_cleanup(self, question: str) -> dict[str, Any]:
        """Get research report from GPT Researcher with proper resource cleanup"""
        researcher = None
        try:
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
            }
        finally:
            # Force cleanup of researcher and any potential connections
            if researcher:
                # Force garbage collection of the researcher instance
                del researcher
            
            # Force garbage collection to help clean up any lingering references
            gc.collect()

    def _run_async_with_new_loop(self, coro, timeout=900):
        """
        Run async coroutine in a new event loop with proper cleanup.
        This prevents event loop resource leaks by ensuring each loop is properly closed.
        """
        loop = None
        try:
            # Create a new event loop for isolation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the coroutine with timeout
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
            
        finally:
            if loop:
                try:
                    # Cancel any pending tasks
                    pending_tasks = asyncio.all_tasks(loop)
                    if pending_tasks:
                        for task in pending_tasks:
                            task.cancel()
                        
                        # Give tasks a moment to cancel gracefully
                        try:
                            loop.run_until_complete(
                                asyncio.wait_for(
                                    asyncio.gather(*pending_tasks, return_exceptions=True),
                                    timeout=2.0
                                )
                            )
                        except asyncio.TimeoutError:
                            # If tasks don't cancel in time, continue with cleanup
                            pass
                    
                    # Close the loop
                    loop.close()
                    
                except Exception:
                    # If cleanup fails, at least try to close the loop
                    try:
                        loop.close()
                    except Exception:
                        pass
                finally:
                    # Clear the event loop from thread-local storage
                    asyncio.set_event_loop(None)
                    
            # Force garbage collection to help clean up event loop resources
            gc.collect()

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
                # Run async research with proper event loop management and cleanup
                start_time = time.time()
                research_results = self._run_async_with_new_loop(
                    self._get_report_with_cleanup(question), 
                    timeout=900
                )
                latency = time.time() - start_time
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
                        "usage": costs,
                        "latency": latency,
                    },
                    actual_queried_message_list=message_list,
                )
            
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="",
                    response_metadata={"usage": None, "error": str(e)},
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
                if trial >= 3:
                    return SamplerResponse(
                        response_text=f"Research failed after {trial} attempts: {str(e)}",
                        response_metadata={"usage": None, "error": str(e)},
                        actual_queried_message_list=message_list,
                    )
