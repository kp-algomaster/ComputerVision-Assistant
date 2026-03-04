"""Paper to Code Agent — standalone agent for scaffolding implementations from papers."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from zeroclaw_tools import create_agent, web_search, file_read, file_write, shell

from cv_agent.config import AgentConfig, load_config
from cv_agent.tools.paper_to_code import (
    scaffold_paper_implementation,
    generate_model_skeleton,
    generate_training_loop,
)
from cv_agent.tools.paper_fetch import fetch_arxiv_paper, search_arxiv
from cv_agent.tools.equation_extract import extract_equations, extract_key_info
from cv_agent.tools.spec_generator import generate_spec

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a specialized Paper to Code Agent for computer vision research.

Your job is to translate research papers into runnable PyTorch implementations.

WORKFLOW for "implement this paper" requests:
1. Call fetch_arxiv_paper to get the full paper details
2. Call extract_equations to capture all mathematical formulations
3. Call extract_key_info with focus="architecture" to understand the model structure
4. Call generate_spec to create a structured implementation spec
5. Call scaffold_paper_implementation to produce the full code scaffold
6. Summarise the generated files and provide next steps for the user

For architecture-specific questions:
- Call generate_model_skeleton with the architecture description
- Call generate_training_loop with the paper's loss function

IMPLEMENTATION PRINCIPLES:
- Faithfully implement the paper's described method
- Prefer PyTorch idiomatic patterns (nn.Module, nn.functional)
- Include shape comments in forward() for every major operation
- Generate test code in __main__ blocks
- Note any missing details from the paper (hyperparameters, implementation tricks)

Always fetch the paper before implementing — never implement from memory.
"""


def _build_tools(config: AgentConfig) -> list:
    return [
        scaffold_paper_implementation,
        generate_model_skeleton,
        generate_training_loop,
        fetch_arxiv_paper,
        search_arxiv,
        extract_equations,
        extract_key_info,
        generate_spec,
        web_search,
        file_read,
        file_write,
        shell,
    ]


async def run_paper_to_code_agent(
    message: str,
    config: AgentConfig | None = None,
    history: list[Any] | None = None,
) -> str:
    """Run the Paper to Code Agent with a user message."""
    if config is None:
        config = load_config()

    if not config.agents.paper_to_code.enabled:
        return "Paper to Code Agent is disabled. Enable it in config under agents.paper_to_code."

    model = config.agents.paper_to_code.model_override or config.llm.model
    tools = _build_tools(config)
    agent = create_agent(
        tools=tools,
        model=model,
        api_key=config.llm.api_key or None,
        base_url=config.llm.base_url,
    )

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=message))

    result = await agent.ainvoke({"messages": messages})
    return result["messages"][-1].content
