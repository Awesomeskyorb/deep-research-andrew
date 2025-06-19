import asyncio
import os
import time
import json
from typing import List, Dict, Any, AsyncIterator, ClassVar
import re
import logging
"""
sub agent for credibility
cover 

calculate tokens per minute
Add a way to find tokens / task

RAG model with database

Brave search

Add a way to see the model's thinking process per research step

mitigate information calls w/ redis
"""

import vertexai
from vertexai import agent_engines


import time


# ADK Imports
from google.adk.agents import Agent, LlmAgent, SequentialAgent, LoopAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService, Session, State
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.tools import ToolContext, google_search
from google.genai.types import Content, Part

from tools import (
    company_profile_tool, price_target_summary_tool, stock_chart_5min_tool,
    income_statement_tool, balance_sheet_tool, cash_flow_tool,
    key_metrics_tool, financial_ratios_tool
)
# ArtifactService is not used in this version for report saving
# from google.adk.artifacts import InMemoryArtifactService, ArtifactService 

# --- Configuration (Replace with your actual keys and preferences) ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyD0miABpu59h8h-aGk4bkAm9S7jHYcE5os" # Replace
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
MODEL_GEMINI_FLASH = "gemini-2.5-pro-preview-06-05"


# Ensure logs directory exists for saving reports
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Tools



#ADD LATEX SUPPORT FOR PDF
def save_report_to_log_file_tool(report_content: str, filename: str, tool_context: ToolContext) -> str:
    """
    Saves the report content to a local .log file.
    Args:
        report_content: The content of the report to save.
        filename: The desired filename (e.g., "report.log").
        tool_context: The context for the tool call (can be used for logging, etc.).
    Returns:
        A string confirming the save operation or an error message.
    """
    print(f"--- TOOL: save_report_to_log_file_tool called for filename: {filename} ---")
    try:
        if not filename.lower().endswith(".log"):
            original_filename = filename
            filename += ".log"
            print(f"Warning: Appended '.log' to filename. Original: '{original_filename}', New: '{filename}'.")
            

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Report generated at: {time.asctime()}\n")
            f.write("="*80 + "\n")
            f.write("Report Title (from filename): " + os.path.basename(filename) + "\n")
            f.write("="*80 + "\n\n")
            f.write(report_content)
            f.write("\n\n" + "="*80 + "\n")
            f.write("End of Report\n")
        
        return f"Report content successfully saved to local file: {os.path.abspath(filename)}"
    except IOError as e:
        print(f"Error saving report to log file '{filename}': {e}")
        return f"Failed to save report to '{filename}'. Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred while saving report to log file '{filename}': {e}")
        return f"Unexpected error saving report to '{filename}'. Error: {e}"

#Agent classes

class PlanningAgent(LlmAgent):
    def __init__(self, model: Any, **kwargs): 
        super().__init__(
            name="PlanningAgent",
            model=model,
            instruction=(
                "You are a strategic research planner tasked with creating an in-depth, multi-step research plan. "
                "For the user's query, generate a detailed series of tasks that cover: context and background, key subtopics, specific data points or metrics to gather, potential sources or references to consult, and follow-up sub-questions for deeper exploration. "
                "Output the plan as a JSON list of strings, where each string is a detailed task description."
                "Available tools include: company_profile_tool, price_target_summary_tool, stock_chart_5min_tool, income_statement_tool, balance_sheet_tool, cash_flow_tool, key_metrics_tool, financial_ratios_tool"
            ),
            description="Generates a multi-step research plan based on user query.",
            output_key="research_plan",
            **kwargs # Pass through any other relevant kwargs
        )

class ResearchStepAgent(LlmAgent):
    def __init__(self, model: Any, **kwargs):
        super().__init__(
            name="ResearchStepAgent",
            model=model,
            tools=[
                   company_profile_tool,
                   price_target_summary_tool,
                   stock_chart_5min_tool,
                   income_statement_tool, 
                   balance_sheet_tool, 
                   cash_flow_tool,
                   key_metrics_tool, 
                   financial_ratios_tool
                   ],
            instruction=(
                "You are an expert financial analysis research agent with deep expertise in equity research, fundamental analysis, technical analysis, and market intelligence. "
                "Your role is to conduct comprehensive, data-driven financial research that provides actionable insights for investment decision-making. "
                "You may be provided with previous research findings from prior steps. Use them to inform your current investigation and avoid re-doing work. "
                "For each research task, conduct a thorough investigation using available tools. "
                "Return your findings as a JSON object with keys: 'task', 'findings', and 'sources' (a list of URLs)."
            ),
            description="Executes a single research task using Google Search. Uses rational decision frameworks to develop a nuanced outlook on the topic.",
            output_key="current_step_findings",
            **kwargs # pass through any other relevant kewyord args
        )

class ResearchOrchestratorAgent(BaseAgent):
    research_step_agent: ResearchStepAgent
    max_research_steps: int = 35 # max research steps 
    max_research_duration: int = 0 # max time in seconds for research phase, 0=no limit
    # name and description are inherited from BaseAgent and will be set at instantiation

    async def run_async(self, context: InvocationContext) -> AsyncIterator[Event]:
        # research orchestration start
        yield Event(author=self.name, content=Content(parts=[Part(text="Starting research orchestration")]))
        print(f"--- AGENT: {self.name} starting ---")
        start_time = time.time() # tracks time so it doesnt go over max duration

        research_plan_str = context.session.state.get("research_plan")
        # markdown fences, extracts json from session state
        if research_plan_str:
            m = re.search(r"(\[.*\])", research_plan_str, flags=re.DOTALL)
            if m:
                research_plan_str = m.group(1)
        all_findings = context.session.state.get("all_research_findings", [])

        if not research_plan_str:
            #if theres no event
            yield Event(
                author=self.name,
                content=Content(parts=[Part(text="No research plan found in state.")])
            )
            return

        try:
            research_plan = json.loads(research_plan_str)
            if not isinstance(research_plan, list):
                raise ValueError("Research plan is not a list.")
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"Error decoding research plan: {e}. Plan was: {research_plan_str}"
            from google.genai.types import Content, Part
            # Emit parsing error as final event for this phase
            yield Event(    
                author=self.name,
                content=Content(parts=[Part(text=error_msg)])
            )
            return

        steps_executed = 0
        for i, task in enumerate(research_plan):
            if self.max_research_duration and (time.time() - start_time) >= self.max_research_duration:
                yield Event(author=self.name, content=Content(parts=[Part(text=f"Reached max research duration of {self.max_research_duration} seconds.")]))
                break
            if steps_executed >= self.max_research_steps: 
                print(f"--- AGENT: {self.name} - Reached max research steps ({self.max_research_steps}) ---")
                break
            print(f"--- AGENT: {self.name} - Starting research task {i+1}: {task} ---")
            # build prompt and invoke sub agents
            research_prompt = (
                f"Perform this research task: {task}. Provide JSON with 'task' and 'findings'."
            )
            # research sub agent exec
            response_event = await self._call_sub_agent_mock(self.research_step_agent, research_prompt, context)
            yield response_event
            step_findings_str = None
            if response_event.actions and response_event.actions.state_delta and "current_step_findings" in response_event.actions.state_delta:
                step_findings_str = response_event.actions.state_delta["current_step_findings"]
            elif response_event.content and response_event.content.parts:
                step_findings_str = response_event.content.parts[0].text
            if step_findings_str:
                try:
                    step_findings_json = json.loads(step_findings_str)
                    if 'task' not in step_findings_json:
                        step_findings_json['task'] = task
                    all_findings.append(step_findings_json)
                    print(f"--- AGENT: {self.name} - Findings for task '{task}': {str(step_findings_json.get('findings','N/A'))[:100]}... ---")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"--- AGENT: {self.name} - Error processing findings for task '{task}': {e}. Findings data: {step_findings_str}")
                    all_findings.append({"task": task, "findings": "Error processing sub-agent output.", "raw_output": str(step_findings_str)})
            else:
                all_findings.append({"task": task, "findings": "No content from research step."})
            steps_executed += 1

        state_delta = {"all_research_findings": all_findings}
        # final state update for this research phase
        yield Event(
            author=self.name,
            content=Content(parts=[Part(text=f"Research orchestration completed. {len(all_findings)} findings collected.")]),
            actions=EventActions(state_delta=state_delta)
        )
        print(f"--- AGENT: {self.name} ending ---")
        # state_delta handles end of research phase

class SynthesisAgent(LlmAgent):
    input_keys: ClassVar[list[str]] = ["all_research_findings"]
    def __init__(self, model: Any, **kwargs): # Changed model_name to model, use generic kwargs
        super().__init__(
            name="SynthesisAgent",
            model=model, # Use model
            instruction=(
                "You will be provided with research findings."
                "Synthesize this information into a single, coherent, well-structured textual report. "
                "The report should have an introduction, a body discussing key findings, and a conclusion. "
                "The input JSON list of findings will be automatically provided from session.state['all_research_findings']."
                "Be as detailed in your final report as possible."
            ),
            description="Synthesizes research findings into a comprehensive report.",
            output_key="synthesized_report_content",
            **kwargs # Pass through any other LlmAgent compatible kwargs
        )

class ReportSavingAgent(LlmAgent):
    input_keys: ClassVar[list[str]] = ["synthesized_report_content"]
    report_filename: ClassVar[str]

    def __init__(self, model: Any, report_filename: str = "deep_research_report.log", **kwargs): # Changed model_name to model
        # Store report_filename as a private attribute before initializing the model
        object.__setattr__(self, "report_filename", report_filename)
        
        # Filter out 'report_filename' from kwargs if it was passed, as it's handled by this class
        llm_kwargs = {k: v for k, v in kwargs.items() if k != 'report_filename'}

        super().__init__(
            name="ReportSavingAgent",
            model=model,
            tools=[save_report_to_log_file_tool],
            instruction=(
                f"You are a report finalization assistant. The synthesized report content is available "
                f"in session.state['synthesized_report_content']. "
                f"Use the 'save_report_to_log_file_tool' to save this content to a local file named '{self.report_filename}'. "
                "Pass the content from 'synthesized_report_content' as the 'report_content' argument and "
                f"'{self.report_filename}' as the 'filename' argument to the tool."
            ),
            description="Saves the final report to a .log file using a tool.",
            **llm_kwargs
        )

# --- Main Sequential Agent for the Trade Arena ---
class LLMTradeArenaAgent(SequentialAgent):
    def __init__(self, planning_agent: PlanningAgent,
                 research_orchestrator: ResearchOrchestratorAgent,
                 synthesis_agent: SynthesisAgent,
                 report_saving_agent: ReportSavingAgent,
                 **kwargs):
        super().__init__(
            name="LLMTradeArenaMainAgent",
            sub_agents=[
                planning_agent,
                research_orchestrator,
                synthesis_agent,
                report_saving_agent
            ],
            description="Main agent orchestrating the LLM Trade Arena deep research workflow.",
            **kwargs
        )

# runner and session start
async def run_trade_arena(query: str, output_log_filename: str = "deep_research_output.log"):
    """Orchestrate planning, research, synthesis, and report saving sequentially using dedicated runners."""
    session_service = InMemorySessionService()
    user_id = "test_user_004"
    session_id = f"{user_id}:{int(time.time())}"
    initial_state: dict[str, Any] = {}
    session = await session_service.create_session(
        app_name="llm_trade_arena_local_log_app",
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )
    print(f"Session created: {session.id} with initial state: {session.state}")

    from google.genai.types import Content, Part
    # 1. Planning Phase
    print("\n--- Planning Phase ---")
    planning_runner = Runner(agent=PlanningAgent(model=MODEL_GEMINI_FLASH),
                             app_name="llm_trade_arena_local_log_app",
                             session_service=session_service)
    plan_msg = Content(role="user", parts=[Part(text=query)])
    research_plan_str = None
    # Debug: print all planning phase events
    async for event in planning_runner.run_async(user_id=user_id, session_id=session.id, new_message=plan_msg):
        print(f"[DEBUG Planning] Author: {event.author}, Type: {type(event).__name__}, Content: {event.content}, Actions: {event.actions}")
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    print(f"  [AGENT TOOL CALL] Function: {part.function_call.name}, Args: {dict(part.function_call.args)}")
                if part.function_response:
                    print(f"  [TOOL RESPONSE] Function: {part.function_response.name}, Response: {part.function_response.response}")
        if event.actions and event.actions.state_delta and "research_plan" in event.actions.state_delta:
            research_plan_str = event.actions.state_delta["research_plan"]
        elif event.content and event.content.parts:
            research_plan_str = event.content.parts[0].text
        if event.is_final_response():
            break
    await asyncio.sleep(5) # Sleep to avoid rate limiting
    if not research_plan_str:
        print("No research plan generated; exiting.")
        return
    # Clean up JSON: strip any markdown fences and extract JSON array
    m = re.search(r"(\[.*\])", research_plan_str, flags=re.DOTALL)
    if m:
        research_plan_str = m.group(1)
    try:
        research_plan = json.loads(research_plan_str)
    except Exception as e:
        print(f"Error parsing research plan: {e}")
        return

    # 2. Research Steps
    print("\n--- Research Steps Phase ---")
    research_runner = Runner(agent=ResearchStepAgent(model=MODEL_GEMINI_FLASH),
                              app_name="llm_trade_arena_local_log_app",
                              session_service=session_service)
    all_findings: list[Any] = []
    # Load existing findings if any from the session state
    if "all_research_findings" in session.state:
        try:
            # The state might store it as a JSON string
            loaded_findings = json.loads(session.state["all_research_findings"])
            if isinstance(loaded_findings, list):
                all_findings = loaded_findings
        except (json.JSONDecodeError, TypeError):
            print("Could not load existing findings, starting fresh.")
            all_findings = []

    for idx, task in enumerate(research_plan, start=1):
        print(f"Task {idx}/{len(research_plan)}: {task}")

        # Construct prompt with previous findings for context
        findings_so_far = json.dumps(all_findings, indent=2)
        prompt = (
            f"Here are the research findings gathered so far:\n{findings_so_far}\n\n"
            f"Your current research task is: {task}\n\n"
            "Please execute this task, taking into account the previous findings to avoid redundant work and build upon existing data. "
            "Return ONLY a single valid JSON object with 'task', 'findings', and 'sources' keys."
        )
        task_msg = Content(role="user", parts=[Part(text=prompt)])

        # Debug: print all research step events
        async for event in research_runner.run_async(user_id=user_id, session_id=session.id, new_message=task_msg):
            print(f"[DEBUG Research] Author: {event.author}, Type: {type(event).__name__}, Content: {event.content}, Actions: {event.actions}")
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        print(f"  [AGENT TOOL CALL] Function: {part.function_call.name}, Args: {dict(part.function_call.args)}")
                    if part.function_response:
                        print(f"  [TOOL RESPONSE] Function: {part.function_response.name}, Response: {part.function_response.response}")

            if event.actions and event.actions.state_delta and "current_step_findings" in event.actions.state_delta:
                new_finding_str = event.actions.state_delta["current_step_findings"]
                try:
                    # Clean markdown fences and other text around JSON
                    m = re.search(r'\{.*\}', new_finding_str, re.DOTALL)
                    if m:
                        cleaned_finding_str = m.group(0)
                        new_finding = json.loads(cleaned_finding_str)
                        if 'task' not in new_finding:
                            new_finding['task'] = task
                        all_findings.append(new_finding)
                    else:
                        # Fallback for non-JSON or malformed output
                        print(f"Warning: Could not find a JSON object in the output for task '{task}'. Storing as text.")
                        all_findings.append({"task": task, "findings": new_finding_str, "sources": []})

                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Error parsing research step JSON for task '{task}': {e}. Raw output: {new_finding_str}")
                    all_findings.append({"task": task, "findings": "Error: Could not parse sub-agent JSON output.", "raw_output": new_finding_str})
                
                # IMPORTANT: Update session state after each step
                session.state["all_research_findings"] = json.dumps(all_findings)

            if event.is_final_response():
                break
        await asyncio.sleep(5) # Sleep to avoid rate limiting
    session.state["all_research_findings"] = json.dumps(all_findings)

    # 3. Synthesis Phase
    print("\n--- Synthesis Phase ---")
    # Initialize synthesis runner
    synth_runner = Runner(agent=SynthesisAgent(model=MODEL_GEMINI_FLASH),
                          app_name="llm_trade_arena_local_log_app",
                          session_service=session_service)
    synthesized_report = None
    # Retry logic in case of model overload
    from google.genai.errors import ServerError
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Debug: print all synthesis phase events
            async for event in synth_runner.run_async(user_id=user_id, session_id=session.id, new_message=Content(role="user", parts=[Part(text="Synthesize findings")] )):
                print(f"[DEBUG Synthesis] Author: {event.author}, Type: {type(event).__name__}, Content: {event.content}, Actions: {event.actions}")
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.function_call:
                            print(f"  [AGENT TOOL CALL] Function: {part.function_call.name}, Args: {dict(part.function_call.args)}")
                        if part.function_response:
                            print(f"  [TOOL RESPONSE] Function: {part.function_response.name}, Response: {part.function_response.response}")
                if event.actions and event.actions.state_delta and "synthesized_report_content" in event.actions.state_delta:
                    synthesized_report = event.actions.state_delta["synthesized_report_content"]
                if event.is_final_response():
                    break
            # Break out if succeeded
            if synthesized_report is not None:
                break
        except ServerError as e:
            print(f"Model overloaded (attempt {attempt}/{max_retries}): {e}. Retrying in {attempt * 5}s...")
            await asyncio.sleep(attempt * 5)
        finally:
            await asyncio.sleep(5) # Sleep to avoid rate limiting
    if synthesized_report is None:
        print(f"Failed to synthesize report after {max_retries} attempts; exiting.")
        return

    # 4. Report Saving Phase (direct file write)
    print("\n--- Report Saving Phase ---")
    # Directly save synthesized report to the specified log file
    confirmation = save_report_to_log_file_tool(synthesized_report, output_log_filename, None)
    print(confirmation)
    print("\nAll phases complete. The report should be in the logs folder.")

# --- Main Execution ---
if __name__ == "__main__":
    user_query = input("Please enter your research query: ")
    if not user_query:
        user_query = "Do comprehensive research on Apple stock and the most likely direction"
        print(f"No query entered, using default: {user_query}")
    
    # Save report log into the centralized logs folder
    # Sanitize query to create a valid filename
    sanitized_query = re.sub(r'[\\/*?:"<>|]', "", user_query)
    log_file_name = os.path.join(LOG_DIR, f"{sanitized_query[:50]}_report.log")

    try:
        asyncio.run(run_trade_arena(query=user_query, output_log_filename=log_file_name))
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
        import traceback
        traceback.print_exc()