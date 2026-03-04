# Computer Vision Assistant 👁️

An autonomous Computer Vision research assistant — monitors arXiv, processes papers, builds knowledge graphs, generates specs, and runs vision tasks locally via Ollama and MLX. Powered by [ZeroClaw](https://github.com/zeroclaw-labs/zeroclaw).

---

## Architecture

```mermaid
graph TD
    UI["🌐 Web UI\n(FastAPI · port 8420)"]
    CLI["⌨️ CLI\ncv-agent"]
    WS["WebSocket\n/ws/chat"]

    ORCH["🧠 Agent Orchestrator\nagent.py · LangGraph ReAct"]
    ZC["⚙️ ZeroClaw\ncreate_agent · @tool"]

    subgraph Tools["🔧 Tools"]
        T1["search_arxiv\nfetch_paper"]
        T2["analyze_image\ndescribe_image"]
        T3["extract_equations\ngenerate_spec"]
        T4["add_to_graph\nquery_graph"]
        T5["shell · file_read\nfile_write · web_search"]
        T6["pull_vision_model\nprobe_hardware"]
    end

    subgraph Backends["⚡ Backends"]
        OL["Ollama\nlocalhost:11434"]
        MLX["MLX\nApple Silicon"]
        AX["ArXiv API"]
        SS["Semantic Scholar"]
        FS["Local Filesystem\n& Obsidian Vault"]
    end

    UI --> WS --> ORCH
    CLI --> ORCH
    ORCH --> ZC --> Tools
    T1 --> AX & SS
    T2 --> OL & MLX
    T3 & T4 --> FS
    T5 --> FS
    T6 --> OL
```

---

## Research → Knowledge Pipeline

```mermaid
flowchart LR
    A["📡 Sources\nArXiv · PWC\nSemantic Scholar"]
    B["📄 Paper Fetch\nAbstract + PDF"]
    C["∑ Extract\nEquations · Arch\nDatasets · Metrics"]
    D["📋 Spec File\nspec.md"]
    E["🕸️ Knowledge Graph\nObsidian Vault"]
    F["📰 Weekly Digest\nMarkdown Blog"]

    A -->|monitor| B
    B -->|parse| C
    C -->|generate| D
    C -->|index| E
    E -->|summarise| F
```

---

## Hardware-Aware Model Selection

```mermaid
flowchart LR
    HW["🖥️ Hardware Probe\nllmfit system"]
    INFO["M4 Max\n36 GB RAM · 36 GB VRAM\n14 cores · Metal"]
    RANK["📊 llmfit rank\nmodels by fit score"]
    SEL["✅ Selected Model\nminimax-m2.5:cloud"]
    AG["🧠 Agent\ncreate_agent(model)"]

    HW --> INFO --> RANK --> SEL --> AG
```

---

## Web UI

Single-page app at `http://localhost:8420` using a sidebar layout inspired by OpenClaw.

![Web UI](docs/screenshots/Web-UI.png)

---

## Skills

Skills are specialised capabilities the agent can perform. A skill is **Ready** when all required powers and packages are available.

| Icon | Skill | Category | Status |
|------|-------|----------|--------|
| ✍️ | Write Research Blog | Content | ✅ Ready |
| 📰 | Weekly Digest | Content | ✅ Ready |
| 📧 | Email Reports | Content | ⚡ Needs Power (Email) |
| 🎥 | Video Understanding | Vision | ⚡ Needs Power (Vid-LLMs) |
| 🔍 | Object Detection | Vision | ⚡ Needs Power (2D Image Processing) |
| 🎯 | Object Tracking | Vision | ⚡ Needs Power (2D Image Processing) |
| ✂️ | Image Segmentation | Vision | ⚡ Needs Power (2D Image Processing) |
| 🧩 | Instance Segmentation | Vision | ⚡ Needs Power (2D Image Processing) |
| 📋 | Paper → Spec | Research | ✅ Ready |
| 🕸️ | Knowledge Graph | Research | ✅ Ready |
| ∑ | Equation Extraction | Research | ✅ Ready |
| 📄 | Document Text Extraction | Research | ⚡ Needs Power (OCR) |
| 🏆 | Kaggle Competition | ML | ⚡ Needs Power (Kaggle) |
| 🎯 | Model Fine-Tuning | ML | ⚡ Needs Power (HuggingFace / Azure ML) |
| 📊 | Dataset Analysis | ML | ✅ Ready |

**6 / 15 skills ready** out of the box. Unlock the rest by configuring the relevant Powers.

---

## Agents

Agents are standalone, focused AI workers — each with its own system prompt, curated tool set, and dedicated WebSocket endpoint (`/ws/agent/<id>`). They can also be invoked by the main agent via delegation tools.

| Icon | Agent | Description | Status |
|------|-------|-------------|--------|
| ✍️ | **Blog Writer** | Writes research blog posts from papers, summaries, or topics. Fetches live paper data before writing. | ✅ Ready |
| 🌐 | **Website Maintenance** | Audits sites for broken links, uptime, and on-page SEO issues. | ✅ Ready |
| 🏋️ | **Model Training** | Generates training configs, cost estimates, and full training scripts for any CV model/task. | ✅ Ready |
| 📊 | **Data Visualization** | Generates matplotlib/plotly chart code and extracts metrics tables from papers. | ✅ Ready |
| 📄→💻 | **Paper to Code** | Scaffolds complete PyTorch implementations from ArXiv papers — model, training loop, dataset class. | ✅ Ready |

Each agent is accessible via:
- **Web UI** — select the agent from the sidebar
- **WebSocket** — `ws://localhost:8420/ws/agent/<id>` (e.g. `blog_writer`, `paper_to_code`)
- **REST** — `GET /api/agents` lists all agents; `GET /api/agents/<id>` returns agent details
- **Main agent delegation** — the main agent auto-delegates tasks using `delegate_<agent>` tools

Per-agent model overrides: set `BLOG_WRITER_MODEL`, `WEBSITE_AGENT_MODEL`, `TRAINING_AGENT_MODEL`, `VIZ_AGENT_MODEL`, or `PAPER_TO_CODE_MODEL` in `.env` to use a different model for a specific agent.

---

## Powers

Powers are external resources and integrations. Active powers unlock additional skills and expand what the agent can do.

### 🔌 Built-in (always available)

| Icon | Power | Status | Notes |
|------|-------|--------|-------|
| 🔍 | Internet Search | ✅ Active | DuckDuckGo by default; set `BRAVE_API_KEY` for higher quality |
| 📁 | Local File System | ✅ Active | `file_read`, `file_write`, `shell` via ZeroClaw |
| 📚 | ArXiv | ✅ Active | Free public API — no key required |
| 🔬 | Semantic Scholar | ⚠️ Limited | Rate-limited; set `SEMANTIC_SCHOLAR_API_KEY` for full access |
| 🖼️ | 2D Image Processing | ✅ Active | Pillow + OpenCV; unlocks Object Detection, Tracking, Segmentation skills |
| 🧊 | 3D Image Processing | 📦 Install | Requires `open3d`; `pip install open3d` |

### 🔗 Integrations (configure in Powers view)

| Icon | Power | Status | Env Var |
|------|-------|--------|---------|
| 📧 | Email (SMTP) | Inactive | `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` |
| 🤗 | HuggingFace Hub | Inactive | `HF_TOKEN` |
| 🏆 | Kaggle | Inactive | `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| 🐙 | GitHub | Inactive | `GITHUB_TOKEN` |
| 🔤 | OCR | Inactive | `OCR_ENGINE` (`tesseract`, `easyocr`, or `monkeyocr`); unlocks Document Text Extraction skill |
| 🎬 | Vid-LLMs | Inactive | `VID_LLM_MODEL` (e.g. `video-llava`, `internvl2`); unlocks Video Understanding skill |

### ☁️ Cloud Compute

| Icon | Power | Status | Env Var |
|------|-------|--------|---------|
| ☁️ | Azure ML | Inactive | `AZURE_SUBSCRIPTION_ID`, `AZURE_ML_WORKSPACE` |
| 🚀 | RunPod | Inactive | `RUNPOD_API_KEY` |

All powers are configurable directly from the **Powers** view in the UI — no manual `.env` editing required.

---

## ZeroClaw Integration

ZeroClaw is the **tool execution layer** between the agent orchestrator and CV tools.

```mermaid
graph LR
    A["🧠 Agent\nLangGraph ReAct"] --> B["⚙️ ZeroClaw\ncreate_agent"]
    B --> C["@tool functions\n(Python decorators)"]
    C --> D["Ollama · MLX\nArXiv · Filesystem\nWeb Search"]

    style A fill:#1a3a5c,stroke:#58a6ff,color:#e6edf3
    style B fill:#2a1f4a,stroke:#bf8fff,color:#e6edf3
    style C fill:#1c2128,stroke:#30363d,color:#e6edf3
    style D fill:#0d1117,stroke:#30363d,color:#8b949e
```

**Current status:** `zeroclaw-tools` is not yet on PyPI. A local shim at `src/zeroclaw_tools/__init__.py` provides the identical API surface via LangChain + LangGraph. When the package ships:

```bash
pip install zeroclaw-tools
rm -rf src/zeroclaw_tools/   # zero other changes required
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) running locally
- macOS Apple Silicon recommended (Metal acceleration via MLX)
- [llmfit](https://github.com/AlexsJones/llmfit) for hardware detection: `brew install llmfit`

### Setup

```bash
git clone https://github.com/kp-algomaster/ComputerVision-Assistant
cd ComputerVision-Assistant

python -m venv .venv
source .venv/bin/activate

# Install the agent + ZeroClaw shim dependencies (LangChain, LangGraph, etc.)
pip install -e ".[dev]"

# Optional: install ZeroClaw when it ships on PyPI (shim is used until then)
# pip install zeroclaw-tools

cp .env.example .env   # add API keys
```

> **ZeroClaw shim:** `zeroclaw-tools` is not yet on PyPI. The repo ships a local compatibility shim at `src/zeroclaw_tools/` that provides the identical `@tool` / `create_agent` API via LangChain + LangGraph. `pip install -e ".[dev]"` installs all shim dependencies automatically. Once the real package is published, replace it with `pip install zeroclaw-tools` and delete the `src/zeroclaw_tools/` directory — no other changes needed.

### Launch

```bash
# Web UI — chat + model management + skills/powers dashboard
source .venv/bin/activate
cv-agent ui
# → http://127.0.0.1:8420

# Or start directly with uvicorn
uvicorn cv_agent.web:create_app --factory --host 127.0.0.1 --port 8420 --app-dir src
```

### CLI Commands

```bash
cv-agent start                                     # interactive terminal agent
cv-agent paper https://arxiv.org/abs/2312.00785 --spec  # process a paper
cv-agent digest --week                             # generate weekly digest
cv-agent vision analyze path/to/image.png          # analyse an image
cv-agent knowledge sync                            # sync knowledge graph
```

---

## Project Structure

```
CV_Zero_Claw_Agent/
├── src/
│   ├── cv_agent/
│   │   ├── agent.py              # Agent orchestrator + LangGraph ReAct loop
│   │   ├── cli.py                # Click CLI entry point
│   │   ├── web.py                # FastAPI server + all API endpoints
│   │   ├── config.py             # Pydantic config (AgentConfig, LlmfitConfig)
│   │   ├── ui/
│   │   │   ├── index.html        # 15-view SPA shell
│   │   │   ├── style.css         # Dark theme (GitHub-inspired)
│   │   │   └── app.js            # View routing, chat WS, all loaders
│   │   ├── tools/
│   │   │   ├── vision.py         # Ollama vision tools
│   │   │   ├── mlx_vision.py     # MLX-accelerated vision (Apple Silicon)
│   │   │   ├── paper_fetch.py    # ArXiv / paper fetching
│   │   │   ├── equation_extract.py   # LaTeX equation extraction
│   │   │   ├── knowledge_graph.py    # Obsidian knowledge graph
│   │   │   ├── spec_generator.py     # Paper → spec.md pipeline
│   │   │   ├── hardware_probe.py     # llmfit integration + Ollama management
│   │   │   └── remote.py             # Telegram / Discord / messaging
│   │   ├── research/
│   │   │   ├── monitor.py        # Source monitoring scheduler
│   │   │   ├── digest.py         # Weekly digest generator
│   │   │   └── sources.py        # ArXiv / PWC / Semantic Scholar config
│   │   └── knowledge/
│   │       ├── graph.py          # Graph core logic
│   │       └── obsidian.py       # Obsidian vault writer
│   └── zeroclaw_tools/
│       └── __init__.py           # ZeroClaw shim (delete when PyPI pkg ships)
├── config/
│   └── agent_config.yaml         # Full agent configuration
├── vault/                        # Obsidian knowledge vault output
├── output/                       # Generated specs and digests
└── .env                          # Secrets (gitignored)
```

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `minimax-m2.5:cloud` | LLM model tag |
| `OLLAMA_VISION_MODEL` | `minimax-m2.5:cloud` | Vision model tag |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `BRAVE_API_KEY` | — | Brave Search (upgrades web search quality) |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Removes rate limits on paper search |
| `HF_TOKEN` | — | HuggingFace Hub access |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | — | Kaggle competition tools |
| `GITHUB_TOKEN` | — | GitHub repo access |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` | — | Email power |
| `VAULT_PATH` | `./vault` | Obsidian vault output path |

Full configuration reference: [`config/agent_config.yaml`](config/agent_config.yaml)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

```
MIT License  Copyright (c) 2026 kp-algomaster
```

You are free to use, modify, and distribute this software for any purpose, including commercial use, with no warranty. Attribution appreciated but not required.

### Third-party notices

| Dependency | License |
|------------|---------|
| [LangChain](https://github.com/langchain-ai/langchain) | MIT |
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT |
| [FastAPI](https://github.com/tiangolo/fastapi) | MIT |
| [Ollama](https://github.com/ollama/ollama) | MIT |
| [llmfit](https://github.com/AlexsJones/llmfit) | Apache 2.0 |
| [MLX](https://github.com/ml-explore/mlx) | MIT |
| [Pydantic](https://github.com/pydantic/pydantic) | MIT |

> **Model licenses** vary by provider. `minimax-m2.5:cloud` and other Ollama-served models are subject to their own upstream licenses. Check the model card on [Ollama Hub](https://ollama.com/library) before commercial use.
