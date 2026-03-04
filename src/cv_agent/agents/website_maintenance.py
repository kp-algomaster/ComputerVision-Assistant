"""Website Maintenance Agent — standalone agent for site health auditing."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from zeroclaw_tools import create_agent, web_search, http_request

from cv_agent._history import trim_history
from cv_agent.config import AgentConfig, load_config
from cv_agent.tools.website_maintenance import check_url_health, audit_links, check_seo_basics

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a specialized Website Maintenance Agent.

Your job is to audit websites for health, broken links, and SEO issues.

WORKFLOW:
1. For any URL provided → call check_url_health to verify reachability
2. For link audits → call audit_links (crawls the page and checks all links)
3. For SEO checks → call check_seo_basics
4. Summarise findings with prioritised action items

REPORTING GUIDELINES:
- Group issues by severity: Critical (broken pages) → Warnings (redirects, missing meta) → Info
- Always include actionable recommendations
- For large sites, focus on the most critical pages first
- Report approximate fix effort (quick win vs. structural change)

Use http_request for any custom HTTP checks beyond the built-in tools.
"""


def _build_tools(config: AgentConfig) -> list:
    return [
        check_url_health,
        audit_links,
        check_seo_basics,
        http_request,
        web_search,
    ]


async def run_website_maintenance_agent(
    message: str,
    config: AgentConfig | None = None,
    history: list[Any] | None = None,
) -> str:
    """Run the Website Maintenance Agent with a user message."""
    if config is None:
        config = load_config()

    if not config.agents.website_maintenance.enabled:
        return "Website Maintenance Agent is disabled. Enable it in config under agents.website_maintenance."

    model = config.agents.website_maintenance.model_override or config.llm.model
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
