"""Blog Writer Agent — standalone agent for writing research blog posts."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from zeroclaw_tools import create_agent, web_search, file_read, file_write

from cv_agent.config import AgentConfig, load_config
from cv_agent.tools.blog_writer import draft_blog_post, format_blog_markdown, save_blog_post
from cv_agent.tools.paper_fetch import fetch_arxiv_paper, search_arxiv

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a specialized Blog Writer Agent for computer vision research.

Your job is to create engaging, accurate, and well-structured blog posts about CV research.

WORKFLOW:
1. If given a paper URL or ID → call fetch_arxiv_paper first to get full details
2. If given a topic → call search_arxiv to find recent relevant papers
3. Call draft_blog_post with the gathered material
4. Call format_blog_markdown to polish the draft
5. Call save_blog_post to persist the result

WRITING GUIDELINES:
- Always ground posts in real paper content fetched via tools
- Lead with why the work matters to practitioners
- Include key equations or architectural insights where relevant
- Keep technical accuracy while remaining readable
- Include a TL;DR at the top

Never write a blog post from memory alone — always fetch current paper data first.
"""


def _build_tools(config: AgentConfig) -> list:
    return [
        fetch_arxiv_paper,
        search_arxiv,
        draft_blog_post,
        format_blog_markdown,
        save_blog_post,
        web_search,
        file_read,
        file_write,
    ]


async def run_blog_writer_agent(
    message: str,
    config: AgentConfig | None = None,
    history: list[Any] | None = None,
) -> str:
    """Run the Blog Writer Agent with a user message."""
    if config is None:
        config = load_config()

    if not config.agents.blog_writer.enabled:
        return "Blog Writer Agent is disabled. Enable it in config under agents.blog_writer."

    model = config.agents.blog_writer.model_override or config.llm.model
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
