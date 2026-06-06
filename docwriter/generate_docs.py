"""
generate_docs.py
================
Multi-agent pipeline that reads LLMlight source code and writes Sphinx .rst files.

Architecture
------------
                    ┌─────────────┐
                    │ Orchestrator│  (large model – plans, coordinates, merges)
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────────┐
   │  Analyst  │   │  Writer   │   │   Reviewer    │
   │  Agent    │   │  Agent    │   │   Agent       │
   │(code→spec)│   │(spec→rst) │   │(rst→feedback) │
   └───────────┘   └───────────┘   └───────────────┘

Roles
-----
- Analyst  : reads raw Python source → extracts structured spec (JSON)
- Writer   : takes spec + examples → writes Sphinx RST prose
- Reviewer : reads draft RST + original source → checks accuracy, flags issues
- Orchestrator: drives the loop, merges feedback into final RST

Models used
-----------
- Orchestrator : gemma-4-26b-a4b-it   (best reasoning)
- Analyst      : qwen/qwen3-coder-30b  (code understanding)
- Writer       : openai/gpt-oss-20b    (prose quality)
- Reviewer     : liquid/lfm2-24b-a2b   (fast, critical eye)
- Fallback     : gemma-4-e4b-it        (small, used for simple subtasks)

Usage
-----
    python generate_docs.py                       # writes all RST files
    python generate_docs.py --page Examples       # single page
    python generate_docs.py --dry-run             # print to stdout only
    python generate_docs.py --max-iterations 2    # cap review loops
"""
import logging

import argparse
import json
import re
import sys
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDPOINT = "http://localhost:1234/v1/chat/completions"

MODELS = {
    "orchestrator": "gemma-4-26b-a4b-it",
    "analyst":      "qwen/qwen3-coder-30b",
    "writer":       "openai/gpt-oss-20b",
    "reviewer":     "liquid/lfm2-24b-a2b",
    "small":        "gemma-4-e4b-it",
}

SOURCE_FILES = {
    "LLMlight.py":   Path("LLMlight.py"),
    "__init__.py":   Path("__init__.py"),
    "memory.py":     Path("memory.py"),
    "sqlite_hnsw.py": Path("sqlite_hnsw.py"),
    "RAG.py":        Path("RAG.py"),
    "utils.py":      Path("utils.py"),
    "examples.py":   Path("examples.py"),
}

OUTPUT_DIR = Path(".")

# RST pages to generate and which source files feed them
RST_PAGES = {
    "Summary": {
        "description": "High-level background: what LLMlight is, goals, output capabilities, schematic overview.",
        "sources": ["__init__.py", "LLMlight.py"],
        "existing_rst": "Summary.rst",
    },
    "Algorithm": {
        "description": (
            "Technical workflow: chunking, embedding, SQLite+HNSW storage, RAG retrieval, "
            "statistical validation. Include get_available_models, memory_init/add/save/search code examples."
        ),
        "sources": ["LLMlight.py", "memory.py", "sqlite_hnsw.py", "RAG.py"],
        "existing_rst": "Algorithm.rst",
    },
    "Examples": {
        "description": (
            "Practical, runnable examples: basic prompt, PDF ingestion, loading existing KB, "
            "summaries, adding/removing chunks, context strategies."
        ),
        "sources": ["examples.py", "LLMlight.py", "__init__.py"],
        "existing_rst": "Examples.rst",
    },
    "Saving_and_Loading": {
        "description": (
            "How to persist (memory_save) and reload (memory_init / memory_load) knowledge bases. "
            "Cover SQLite default, ANN index rebuilding, memory_remove, memory_chunks."
        ),
        "sources": ["LLMlight.py", "memory.py", "sqlite_hnsw.py"],
        "existing_rst": "Saving_and_Loading.rst",
    },
    "Installation": {
        "description": (
            "conda env creation, pip install, github install, optional deps table "
            "(sentence-transformers, hnswlib, memvid, distfit, llama-cpp-python), uninstall."
        ),
        "sources": ["__init__.py"],
        "existing_rst": "Installation.rst",
    },
}

# ---------------------------------------------------------------------------
# LLM call primitive
# ---------------------------------------------------------------------------

def _call_llm(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    endpoint: str = ENDPOINT,
) -> str:
    """Single LLM call. Returns response text or raises on error."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        # Strip <think>…</think> blocks (some models emit these)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as exc:
        raise RuntimeError(f"LLM call failed [{model}]: {exc}") from exc


# ---------------------------------------------------------------------------
# Utility: load source files
# ---------------------------------------------------------------------------

def _load_sources(names: list[str], base: Path = Path(".")) -> dict[str, str]:
    out = {}
    for name in names:
        path = SOURCE_FILES.get(name)
        if path is None:
            continue
        full = base / path
        if full.exists():
            out[name] = full.read_text(encoding="utf-8", errors="ignore")
        else:
            out[name] = f"# FILE NOT FOUND: {full}"
    return out


def _load_existing_rst(name: str, base: Path = Path(".")) -> str:
    path = base / name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


# ---------------------------------------------------------------------------
# Agent 1 – Analyst
#   Input : raw source code
#   Output: structured JSON spec (sections, params, methods, examples)
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = """\
You are a senior Python developer and technical writer specialising in documenting LLM libraries.
Your task is to analyse Python source code and extract a STRUCTURED SPECIFICATION in JSON format
that a documentation writer can use to author Sphinx RST pages.

Output ONLY valid JSON – no markdown fences, no preamble, no explanation.

The JSON schema:
{
  "page_title": "str",
  "sections": [
    {
      "title": "str",
      "purpose": "str",
      "key_classes_or_functions": ["str"],
      "key_params": [{"name": "str", "type": "str", "default": "str", "description": "str"}],
      "notes": "str",
      "code_example": "str"   // minimal but correct Python snippet
    }
  ],
  "important_notes": ["str"],
  "backend_notes": "str"
}
"""


def analyst_agent(
    page_name: str,
    page_description: str,
    sources: dict[str, str],
    model: str = MODELS["analyst"],
) -> dict:
    """Extract structured spec from source code for a given RST page."""
    source_block = "\n\n".join(
        f"### FILE: {fname}\n{code[:8000]}"   # cap per file to avoid overflow
        for fname, code in sources.items()
    )
    user = f"""\
RST page to document: {page_name}
Page description / scope: {page_description}

SOURCE CODE:
{source_block}

Extract the documentation spec for this page. Output ONLY valid JSON.
"""
    raw = _call_llm(model, ANALYST_SYSTEM, user, temperature=0.2, max_tokens=3000)

    # Robust JSON extraction
    raw = raw.strip()
    # Remove ```json fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        # Return a minimal fallback so the pipeline can continue
        return {"page_title": page_name, "sections": [], "raw_analyst_output": raw}


# ---------------------------------------------------------------------------
# Agent 2 – Writer
#   Input : JSON spec + existing RST (for style reference)
#   Output: full Sphinx RST text
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """\
You are an expert technical writer producing Sphinx reStructuredText (RST) documentation
for a Python library called LLMlight.

RULES:
- Use correct RST syntax: section headings underlined with # = - ^ " (in that hierarchy).
- Code examples go inside  .. code-block:: python  directives.
- Parameter tables use  .. list-table::  or description-list style.
- Do NOT include .. toctree:: or page-level metadata.
- End every page with  .. include:: add_bottom.add  on its own line.
- Be concise but complete. Prose is professional, not marketing fluff.
- All code examples must be self-contained and correct for LLMlight's actual API.
  Key facts:
    * Default memory backend is SQLite (not FAISS, not MemVid).
    * memory_init uses  store_path=  (NOT file_path=).
    * memory.search() returns list of (id, score, metadata_dict) tuples.
    * memory_remove(ids=…) or memory_remove(query=…).
    * memory_reindex() rebuilds the HNSW ANN index.
    * Models: 'mistralai/mistral-small-3.2', 'gemma-4-26b-a4b-it', etc.

Output ONLY the raw RST text. No preamble, no markdown fences.
"""


def writer_agent(
    page_name: str,
    spec: dict,
    existing_rst: str = "",
    model: str = MODELS["writer"],
) -> str:
    """Turn a structured spec into Sphinx RST."""
    spec_json = json.dumps(spec, indent=2)
    style_hint = (
        f"\nFor style reference (do NOT copy verbatim – write fresh):\n{existing_rst[:2000]}"
        if existing_rst
        else ""
    )
    user = f"""\
Write the Sphinx RST documentation page for: {page_name}

SPECIFICATION (JSON):
{spec_json}
{style_hint}

Output the complete RST document. Start with the page title underlined with #.
"""
    return _call_llm(model, WRITER_SYSTEM, user, temperature=0.4, max_tokens=4096)


# ---------------------------------------------------------------------------
# Agent 3 – Reviewer
#   Input : draft RST + original source snippets
#   Output: feedback JSON {issues: [...], approved: bool, revised_rst: str|null}
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM = """\
You are a meticulous technical reviewer checking Sphinx RST documentation against Python source code.

Your job:
1. Check every code example for correctness against the source code.
2. Check all parameter names, types, and defaults.
3. Check all method names (e.g. memory_init uses store_path=, not file_path=).
4. Identify missing important functionality.
5. Flag RST syntax errors.

Output JSON only:
{
  "approved": true|false,
  "issues": ["issue description", ...],
  "revised_rst": "full corrected RST string OR null if approved"
}

If there are only minor issues, set approved=true and revised_rst=null.
If there are significant errors, set approved=false and provide the full corrected RST in revised_rst.
Output ONLY valid JSON.
"""


def reviewer_agent(
    page_name: str,
    draft_rst: str,
    sources: dict[str, str],
    model: str = MODELS["reviewer"],
) -> dict:
    """Review draft RST for accuracy. Returns dict with approved/issues/revised_rst."""
    source_snippet = "\n\n".join(
        f"### {fname}\n{code[:4000]}"
        for fname, code in sources.items()
    )
    user = f"""\
Page: {page_name}

DRAFT RST:
{draft_rst}

SOURCE CODE (reference):
{source_snippet}

Review the RST. Output JSON only.
"""
    raw = _call_llm(model, REVIEWER_SYSTEM, user, temperature=0.1, max_tokens=4096)
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    # Fallback: assume approved, no issues
    return {"approved": True, "issues": [], "revised_rst": None}


# ---------------------------------------------------------------------------
# Agent 4 – Orchestrator
#   Drives the pipeline per page, runs review loop, writes final file
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """\
You are the orchestration agent for an RST documentation pipeline.
Your task: given a reviewer's feedback and the current RST draft, produce a single
improved RST document that resolves all reviewer issues.

RULES:
- Preserve correct sections from the draft.
- Fix only what the reviewer flagged.
- Keep correct Sphinx RST syntax.
- End with  .. include:: add_bottom.add
- Output ONLY the raw RST text, no preamble.
"""


def orchestrator_merge(
    page_name: str,
    draft_rst: str,
    review: dict,
    model: str = MODELS["orchestrator"],
) -> str:
    """Merge reviewer feedback into an improved RST draft."""
    if review.get("revised_rst"):
        # Reviewer already produced a corrected version — use it as base
        draft_rst = review["revised_rst"]

    issues = review.get("issues", [])
    if not issues:
        return draft_rst

    issues_text = "\n".join(f"- {i}" for i in issues)
    user = f"""\
Page: {page_name}

CURRENT RST DRAFT:
{draft_rst}

REVIEWER ISSUES TO FIX:
{issues_text}

Output the fully corrected RST document.
"""
    return _call_llm(model, ORCHESTRATOR_SYSTEM, user, temperature=0.2, max_tokens=4096)


# ---------------------------------------------------------------------------
# Pipeline: one RST page
# ---------------------------------------------------------------------------

def generate_rst_page(
    page_name: str,
    config: dict,
    source_base: Path = Path("."),
    max_iterations: int = 2,
    verbose: bool = True,
) -> str:
    """Run the full multi-agent pipeline for one RST page. Returns final RST text."""

    def log(msg):
        if verbose:
            print(f"[{page_name}] {msg}", flush=True)

    # Load sources
    sources = _load_sources(config["sources"], source_base)
    existing_rst = _load_existing_rst(config.get("existing_rst", ""), source_base)

    log(f"Loaded {len(sources)} source files.")

    # --- Step 1: Analyst ---
    log(f"Analyst ({MODELS['analyst']}) extracting spec...")
    spec = analyst_agent(page_name, config["description"], sources)
    log(f"Spec sections: {[s.get('title','?') for s in spec.get('sections', [])]}")

    # --- Step 2: Writer ---
    log(f"Writer ({MODELS['writer']}) drafting RST...")
    draft = writer_agent(page_name, spec, existing_rst)
    log(f"Draft length: {len(draft)} chars")

    # --- Step 3+4: Review loop ---
    final = draft
    for iteration in range(max_iterations):
        log(f"Reviewer ({MODELS['reviewer']}) iteration {iteration + 1}...")
        review = reviewer_agent(page_name, final, sources)

        issues = review.get("issues", [])
        approved = review.get("approved", True)

        if issues:
            log(f"Issues found: {len(issues)}")
            for i, issue in enumerate(issues[:5], 1):
                log(f"  {i}. {issue}")
        else:
            log("No issues found.")

        if approved and not issues:
            log("Approved.")
            break

        log(f"Orchestrator ({MODELS['orchestrator']}) merging feedback...")
        final = orchestrator_merge(page_name, final, review)
        log(f"Revised length: {len(final)} chars")

    return final


# ---------------------------------------------------------------------------
# API / Core Pipeline
# ---------------------------------------------------------------------------

def main(
    page=None,
    source_dir=".",
    output_dir=".",
    max_iterations=2,
    dry_run=False,
    endpoint=None,
    orchestrator_model=None,
    analyst_model=None,
    writer_model=None,
    reviewer_model=None,
):
    """
    Programmatic API entry point for the Multi-agent Sphinx RST generator.
    """
    global ENDPOINT
    
    # Apply model overrides if provided
    if orchestrator_model:  MODELS["orchestrator"] = orchestrator_model
    if analyst_model:       MODELS["analyst"] = analyst_model
    if writer_model:        MODELS["writer"] = writer_model
    if reviewer_model:      MODELS["reviewer"] = reviewer_model
    if endpoint:            ENDPOINT = endpoint

    source_base = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pages = (
        {page: RST_PAGES[page]}
        if page
        else RST_PAGES
    )

    results = {}
    for page_name, config in pages.items():
        print(f"\n{'='*60}")
        print(f"   Generating: {page_name}")
        print(f"{'='*60}")
        try:
            rst_text = generate_rst_page(
                page_name=page_name,
                config=config,
                source_base=source_base,
                max_iterations=max_iterations,
                verbose=True,
            )
            results[page_name] = rst_text

            if dry_run:
                print(f"\n--- {page_name}.rst ---\n")
                print(rst_text)
            else:
                out_file = output_path / f"{page_name}.rst"
                out_file.write_text(rst_text, encoding="utf-8")
                print(f"\n   Written: {out_file}  ({len(rst_text)} chars)")

        except Exception as exc:
            print(f"\n   ERROR generating {page_name}: {exc}", file=sys.stderr)
            results[page_name] = None

    # Summary
    print(f"\n{'='*60}")
    print("   Pipeline complete")
    print(f"{'='*60}")
    for name, rst in results.items():
        status = f"{len(rst)} chars" if rst else "FAILED"
        print(f"   {name:<30} {status}")
        
    return results


# ---------------------------------------------------------------------------
# CLI Execution Catch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-agent Sphinx RST generator for LLMlight"
    )
    parser.add_argument(
        "--page",
        choices=list(RST_PAGES.keys()),
        default=None,
        help="Generate a single page (default: all pages)",
    )
    parser.add_argument(
        "--source-dir",
        default=".",
        help="Directory containing LLMlight source files (default: cwd)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write .rst files (default: cwd)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Max review-loop iterations per page (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated RST to stdout instead of writing files",
    )
    parser.add_argument(
        "--endpoint",
        default=ENDPOINT,
        help=f"LLM API endpoint (default: {ENDPOINT})",
    )
    parser.add_argument(
        "--orchestrator-model", default=MODELS["orchestrator"],
        help="Override orchestrator model"
    )
    parser.add_argument(
        "--analyst-model", default=MODELS["analyst"],
        help="Override analyst model"
    )
    parser.add_argument(
        "--writer-model", default=MODELS["writer"],
        help="Override writer model"
    )
    parser.add_argument(
        "--reviewer-model", default=MODELS["reviewer"],
        help="Override reviewer model"
    )
    
    args = parser.parse_args()

    # Unpack the parsed CLI arguments directly into the main API function
    main(**vars(args))