import { renderMermaidSVG, THEMES } from '../node_modules/beautiful-mermaid/dist/index.js'
import { writeFileSync, mkdirSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const OUT = join(__dirname, '..', 'docs', 'diagrams')
mkdirSync(OUT, { recursive: true })

const theme = THEMES['catppuccin-mocha']

function save(name, diagram) {
  const svg = renderMermaidSVG(diagram, { theme })
  const path = join(OUT, `${name}.svg`)
  writeFileSync(path, svg)
  console.log(`  wrote ${path}`)
}

// ── 1. System Architecture ──────────────────────────────────────────────────
save('architecture', `
flowchart TB
  subgraph UI["Web UI  •  :8420"]
    direction LR
    CHAT["💬 Chat"]
    VAULT["📚 Vault Viewer"]
    SPECS["📄 Spec Viewer"]
    DIGESTS["📰 Digest Viewer"]
  end

  subgraph AGENT["CV Zero Claw Agent"]
    direction TB
    ORCH["Agent Orchestrator\nlangchain + langgraph"]
    HW["🔍 Hardware Probe\nllmfit"]
    ORCH <--> HW
  end

  subgraph TOOLS["Tool Layer"]
    direction LR
    VIS["👁️ Vision\nOllama / MLX"]
    PAPER["📥 Paper Fetch\nArXiv / PWC"]
    EQ["∑ Equation\nExtractor"]
    KG["🕸️ Knowledge\nGraph"]
    SPEC["📋 Spec\nGenerator"]
  end

  subgraph MODELS["Model Layer"]
    direction LR
    OLLAMA["Ollama\nqwen2.5-vl · llava"]
    MLX["MLX\nApple Silicon"]
    OLLAMA2["Ollama\nqwen2.5-coder (LLM)"]
  end

  subgraph DATA["Data Sources"]
    direction LR
    ARXIV["ArXiv"]
    PWC["Papers With Code"]
    SS["Semantic Scholar"]
  end

  UI -- "WebSocket /ws/chat" --> AGENT
  AGENT --> TOOLS
  TOOLS --> MODELS
  TOOLS --> DATA
  KG --> VAULT_FS[("Obsidian Vault\n.md files")]
  SPEC --> SPECS_FS[("output/specs/\n*.md")]
`)

// ── 2. Research → Knowledge Pipeline ────────────────────────────────────────
save('research_pipeline', `
flowchart LR
  A(["🔎 Discover\nArXiv / PWC / S2"]) --> B["📥 Fetch Paper\nPDF + metadata"]
  B --> C["∑ Extract\nEquations + Arch"]
  C --> D["📋 Generate\nspec.md"]
  C --> E["🕸️ Add to\nKnowledge Graph"]
  E --> F(["📚 Obsidian\nVault"])
  D --> G(["output/specs/"])
  B --> H["👁️ Vision Model\nFigure Analysis"]
  H --> C
  E --> I["📰 Weekly\nDigest"]
  I --> J(["output/digests/"])
`)

// ── 3. Hardware-Aware Model Selection ───────────────────────────────────────
save('model_selection', `
flowchart TD
  START(["Agent Startup"]) --> PROBE["🔍 llmfit\nHardware Probe"]
  PROBE --> HW["Detect: RAM · VRAM\nCPU · Acceleration"]
  HW --> SCORE["Score ~200 Models\nperfect / good / marginal"]
  SCORE --> BEST["Select Best Fit\nfor multimodal + general"]
  BEST --> UPD["Update Config\nllm.model + vision model"]
  UPD --> BUILD["Build Agent\nwith optimal models"]
  PROBE -- "llmfit not\ninstalled" --> FALLBACK["Use .env /\nconfig defaults"]
  FALLBACK --> BUILD
  BUILD --> TOOL["check_runnable_models\ntool available mid-session"]
`)

console.log('\nAll diagrams generated in docs/diagrams/')
