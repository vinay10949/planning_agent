from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from rich.console import Console

from planning_agent.config import settings
from planning_agent.models import Plan, Replan, PlanExecuteState

console = Console()

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.lm_studio_model,
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        temperature=0.0,
        max_tokens=4096,
    )

llm = get_llm()

# ─── DEFINE TOOLS (Matches Packt Chapter 5 Tool-Calling Executor) ──
# LM Studio supports tool calling for Qwen, Llama, and Gemma!
@tool
def fake_web_search(query: str) -> str:
    """Useful for when you need to ask with search to find factual information."""
    # In a real app, swap this with TavilySearchResults or similar
    return f"Search results for: {query}. Paris is the capital of France. Berlin is the capital of Germany."

@tool
def calculator(expression: str) -> str:
    """Useful for when you need to answer questions about math or compute numbers."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {e}"

tools = [fake_web_search, calculator]

# ─── Prompts & Chains ──────────────────────────────────
# Prompts are deliberately concise to avoid truncated JSON output from local LLMs.
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a planning agent. For the given objective, create a short step-by-step plan. "
               "Rules: Return 3-5 steps. Each step must be ONE short sentence (under 15 words). "
               "Be concise and direct. Do NOT explain or elaborate."),
    ("human", "{objective}"),
])
planner = planner_prompt | llm.with_structured_output(Plan)

# ─── EXECUTOR: ReAct Agent with Tools ──────────────────
# The book implements the executor as an agent that can use tools
# rather than a basic LLM chain, so it can actually perform steps.
executor_agent = create_react_agent(llm, tools)

async def execute_step(state: PlanExecuteState) -> dict:
    step = state["plan"][0]
    console.print(f"[bold yellow]⚡ EXECUTOR: Running step with tools → {step}[/bold yellow]")

    agent_input = {
        "messages": [("human", f"Overall Objective: {state['input']}\n\nExecute this specific step now: {step}")]
    }

    # Invoke the tool-calling agent
    result = await executor_agent.ainvoke(agent_input)

    # Extract the final AI message from the ReAct agent's output
    final_message = result["messages"][-1].content

    return {"past_steps": [(step, final_message)], "plan": state["plan"][1:]}


# ─── Replanner ────────────────────────────────────────
replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a re-planning agent. Evaluate the original objective and past steps. "
               "If the objective is met, return the final answer in feedback and an EMPTY next_steps list. "
               "If not met, return remaining steps. Keep feedback and steps SHORT (under 20 words each)."),
    ("human", "Objective: {objective}\n\nOriginal plan:\n{plan}\n\nSteps completed:\n{past_steps}\n\nRe-evaluate:"),
])
replanner = replanner_prompt | llm.with_structured_output(Replan)

# ─── Graph Nodes ───────────────────────────────────────
MAX_RETRIES = 3

async def plan_step(state: PlanExecuteState) -> dict:
    console.print("[bold blue]📝 PLANNER: Creating initial plan...[/bold blue]")
    for attempt in range(MAX_RETRIES):
        try:
            plan_output = await planner.ainvoke({"objective": state["input"]})
            if plan_output.steps:
                return {"plan": plan_output.steps}
        except Exception as e:
            console.print(f"[bold red]⚠️  Planner attempt {attempt + 1} failed: {e}[/bold red]")
            if attempt == MAX_RETRIES - 1:
                # Fallback: treat the whole task as a single step
                console.print("[bold red]Falling back to single-step plan.[/bold red]")
                return {"plan": [state["input"]]}
    return {"plan": [state["input"]]}

async def replan_step(state: PlanExecuteState) -> dict:
    console.print("[bold magenta]🔄 RE-PLANNER: Evaluating progress...[/bold magenta]")
    past_steps_text = "\n".join(f"Step: {s}\nResult: {r}" for s, r in state["past_steps"])
    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"]))

    for attempt in range(MAX_RETRIES):
        try:
            output = await replanner.ainvoke({
                "objective": state["input"], "plan": plan_text, "past_steps": past_steps_text
            })
            if output.next_steps:
                return {"plan": output.next_steps}
            console.print("[bold green]✅ Task complete![/bold green]")
            return {"response": output.feedback, "plan": []}
        except Exception as e:
            console.print(f"[bold red]⚠️  Replanner attempt {attempt + 1} failed: {e}[/bold red]")
            if attempt == MAX_RETRIES - 1:
                # If replanner keeps failing, treat the last past_step result as the answer
                if state["past_steps"]:
                    last_result = state["past_steps"][-1][1]
                    return {"response": last_result, "plan": []}
                return {"response": "Unable to complete task.", "plan": []}

def should_end(state: PlanExecuteState) -> str:
    if state.get("response") or not state.get("plan"):
        return "end"
    return "continue"

# ─── Build Graph ───────────────────────────────────────
workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", plan_step)
workflow.add_node("executor", execute_step)
workflow.add_node("replanner", replan_step)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replanner")
workflow.add_conditional_edges("replanner", should_end, {"continue": "executor", "end": END})

app = workflow.compile()
