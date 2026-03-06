# Adding Tools (Skills) to Agents

This guide explains how to create a new tool and wire it up to the main agent or a sub-agent in CV Zero Claw Agent.

---

## Architecture overview

```
User message
    └─► Main agent  (agent.py → build_tools)
            ├─► @tool functions         — direct skills (tools/)
            └─► delegate_* tools        — sub-agent wrappers
                    └─► Sub-agent       (agents/<name>.py → _build_tools)
                                └─► @tool functions  — sub-agent skills
```

All tools are LangChain `@tool`-decorated callables. The `zeroclaw_tools` shim re-exports `tool` from `langchain_core.tools` and handles both native function-calling models and text-based JSON tool calls transparently.

---

## Step 1 — Write the tool

Create a file in `src/cv_agent/tools/` (or add to an existing one):

```python
# src/cv_agent/tools/my_tool.py
from zeroclaw_tools import tool

@tool
def my_tool(query: str, max_results: int = 5) -> str:
    """One-line description the LLM uses to decide when to call this.

    Args:
        query: The search query or input.
        max_results: How many results to return.

    Returns:
        A string result the agent can read.
    """
    # implementation
    return f"Result for {query}"
```

**Rules:**
- The function docstring **is** the tool description — write it for the LLM, not a human.
- Parameters must have type annotations; these become the JSON schema the LLM fills in.
- Return a plain `str`. If returning structured data, serialize it to a readable string.
- Async is supported — `async def my_tool(...)` works without any changes.
- Only use MIT / Apache 2.0 / BSD-2/3 licensed libraries (see `CLAUDE.md`).

---

## Step 2 — Provide the tool to an agent

### Option A: Main agent (available to all conversations)

Edit `src/cv_agent/agent.py`:

```python
# 1. Import at the top
from cv_agent.tools.my_tool import my_tool

# 2. Add to build_tools()
def build_tools(config: AgentConfig) -> list:
    tools = [
        ...
        my_tool,   # ← add here
    ]
    return tools
```

Also add it to `SYSTEM_PROMPT` under "Capabilities" so the LLM knows when to use it:

```python
SYSTEM_PROMPT = """
...
8. **My Feature**: Use `my_tool` to do X when the user asks about Y.
...
"""
```

### Option B: Specific sub-agent only

Edit the relevant agent file in `src/cv_agent/agents/`, e.g. `blog_writer.py`:

```python
from cv_agent.tools.my_tool import my_tool

def _build_tools(config: AgentConfig) -> list:
    return [
        ...
        my_tool,   # ← add here
    ]
```

Sub-agents should only receive the tools they actually need — keep the list focused.

---

## Step 3 — (Optional) Add a new sub-agent and delegate it

If the tool warrants its own focused agent:

### 3a. Create the agent file

```python
# src/cv_agent/agents/my_agent.py
from langchain_core.messages import HumanMessage, SystemMessage
from zeroclaw_tools import create_agent
from cv_agent.config import AgentConfig, load_config
from cv_agent.tools.my_tool import my_tool

SYSTEM_PROMPT = "You are a specialized agent for X. Always call my_tool first."

def _build_tools(config: AgentConfig) -> list:
    return [my_tool]

async def run_my_agent(message: str, config: AgentConfig | None = None) -> str:
    if config is None:
        config = load_config()
    tools = _build_tools(config)
    agent = create_agent(
        tools=tools,
        model=config.llm.model,
        api_key=config.llm.api_key or None,
        base_url=config.llm.base_url,
    )
    result = await agent.ainvoke({"messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message),
    ]})
    return result["messages"][-1].content
```

### 3b. Register it in `src/cv_agent/agents/__init__.py`

```python
from cv_agent.agents.my_agent import run_my_agent

__all__ = [..., "run_my_agent"]

AGENT_REGISTRY["my_agent"] = {
    "name": "My Agent",
    "description": "Short description shown in the UI.",
    "icon": "🔍",
    "runner": run_my_agent,
    "config_key": "my_agent",
}
```

### 3c. Add a delegation tool in `agent.py`

In `_make_delegation_tools()`:

```python
if config.agents.my_agent.enabled:
    @tool
    def delegate_my_agent(task: str) -> str:
        """Delegate X tasks to the My Agent.

        Use when the user asks about Y or Z.
        Args:
            task: The task description.
        """
        return asyncio.run(run_my_agent(task, config))
    delegation.append(delegate_my_agent)
```

### 3d. Add the config key

In `config/agent_config.yaml` under `agents:`:

```yaml
agents:
  my_agent:
    enabled: true
    model_override: ""   # leave blank to use the global llm.model
```

And add the corresponding Pydantic model in `src/cv_agent/config.py`.

---

## Quick reference

| What you want | Where to edit |
|---|---|
| New tool available to main agent | `tools/<name>.py` + `agent.py:build_tools` |
| New tool for one sub-agent only | `tools/<name>.py` + `agents/<agent>.py:_build_tools` |
| New sub-agent the main agent can delegate to | `agents/<name>.py` + `agents/__init__.py` + `agent.py:_make_delegation_tools` |
| Enable/disable an agent at runtime | `config/agent_config.yaml` under `agents:` |
| Built-in tools (shell, web_search, etc.) | `src/zeroclaw_tools/__init__.py` |

---

## Tool docstring format

The LLM reads the docstring to decide when and how to call a tool. Follow this format:

```python
@tool
def my_tool(param_a: str, param_b: int = 10) -> str:
    """Short sentence describing what this tool does and when to use it.

    Args:
        param_a: What this parameter represents.
        param_b: What this parameter controls (default: 10).

    Returns:
        What the return value contains.
    """
```

Keep the first line concise — it's used as the one-line hint in the text-mode tool prompt injected by the shim (`zeroclaw_tools._build_tool_prompt`).
