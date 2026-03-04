"""Model Training Agent — standalone agent for training config and script generation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from zeroclaw_tools import create_agent, web_search, file_read, file_write, shell

from cv_agent._history import trim_history
from cv_agent.config import AgentConfig, load_config
from cv_agent.tools.model_training import (
    generate_training_config,
    estimate_training_cost,
    scaffold_training_script,
)
from cv_agent.tools.paper_fetch import fetch_arxiv_paper, search_arxiv
from cv_agent.tools.hardware_probe import check_runnable_models, list_available_models

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a specialized Model Training Agent for computer vision.

Your job is to help users set up, configure, and run CV model training pipelines.

CAPABILITIES:
- Generate training configurations (YAML) for any CV architecture and task
- Estimate GPU hours and cloud costs before committing to a run
- Scaffold complete training scripts (PyTorch, PyTorch Lightning, HuggingFace)
- Check local hardware to recommend feasible models via check_runnable_models
- Fetch paper training details to replicate published results

WORKFLOW for "train a model" requests:
1. Clarify: model, task, dataset, hardware constraints
2. Call check_runnable_models to assess local hardware
3. If replicating a paper → fetch_arxiv_paper for exact hyperparameters
4. Call generate_training_config to produce the YAML
5. Call estimate_training_cost so user knows what to expect
6. Call scaffold_training_script to produce runnable code

Always check hardware constraints before recommending large models.
Never guess hyperparameters — fetch them from the paper when available.
"""


def _build_tools(config: AgentConfig) -> list:
    return [
        generate_training_config,
        estimate_training_cost,
        scaffold_training_script,
        check_runnable_models,
        list_available_models,
        fetch_arxiv_paper,
        search_arxiv,
        web_search,
        file_read,
        file_write,
        shell,
    ]


async def run_model_training_agent(
    message: str,
    config: AgentConfig | None = None,
    history: list[Any] | None = None,
) -> str:
    """Run the Model Training Agent with a user message."""
    if config is None:
        config = load_config()

    if not config.agents.model_training.enabled:
        return "Model Training Agent is disabled. Enable it in config under agents.model_training."

    model = config.agents.model_training.model_override or config.llm.model
    tools = _build_tools(config)
    agent = create_agent(
        tools=tools,
        model=model,
        api_key=config.llm.api_key or None,
        base_url=config.llm.base_url,
    )

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    if history:
        messages.extend(trim_history(list(history), config.cache.max_history_chars))
    messages.append(HumanMessage(content=message))

    result = await agent.ainvoke({"messages": messages})
    return result["messages"][-1].content
