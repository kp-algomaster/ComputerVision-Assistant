"""Data Visualization Agent — standalone agent for generating charts and extracting metrics."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from zeroclaw_tools import create_agent, web_search, file_read, file_write

from cv_agent._history import trim_history
from cv_agent.config import AgentConfig, load_config
from cv_agent.tools.data_visualization import (
    generate_plot_code,
    extract_paper_metrics,
    save_plot_script,
)
from cv_agent.tools.paper_fetch import fetch_arxiv_paper
from cv_agent.tools.equation_extract import extract_key_info

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a specialized Data Visualization Agent for computer vision research.

Your job is to generate publication-quality visualizations and extract structured data from papers.

CAPABILITIES:
- Generate matplotlib/plotly/seaborn chart code for any described dataset
- Extract quantitative results tables from paper text
- Fetch and parse paper data directly from ArXiv

WORKFLOW for "visualize results" requests:
1. If given a paper → call fetch_arxiv_paper to get the full text
2. Call extract_paper_metrics to pull out tables and numbers
3. Call generate_plot_code with the extracted data to produce a chart
4. Call save_plot_script to persist the code
5. Provide a summary of what was visualized and how to run the script

CHART SELECTION GUIDE:
- Comparing models across benchmarks → bar chart or radar chart
- Performance vs. parameter count → scatter plot
- Training curves → line chart
- Confusion matrix → heatmap
- Result distribution → box plot

Always produce runnable code with embedded sample data.
"""


def _build_tools(config: AgentConfig) -> list:
    return [
        generate_plot_code,
        extract_paper_metrics,
        save_plot_script,
        fetch_arxiv_paper,
        extract_key_info,
        web_search,
        file_read,
        file_write,
    ]


async def run_data_visualization_agent(
    message: str,
    config: AgentConfig | None = None,
    history: list[Any] | None = None,
) -> str:
    """Run the Data Visualization Agent with a user message."""
    if config is None:
        config = load_config()

    if not config.agents.data_visualization.enabled:
        return "Data Visualization Agent is disabled. Enable it in config under agents.data_visualization."

    model = config.agents.data_visualization.model_override or config.llm.model
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
