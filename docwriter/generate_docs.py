"""
generate_docs.py
================
Generic multi-agent pipeline that reads any Python project's source code
and writes Sphinx .rst documentation files.

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

Page discovery (no static RST_PAGES required)
----------------------------------------------
1. AST scan (no LLM): extract module docstrings, class names, public functions
   from every .py file → compact project summary.
2. One LLM call (small model): DOC_STRUCTURE_PROMPT + project summary
   → JSON list of {page_title, filename, description, sources}.
3. Optional override: if rst_pages.json exists in --source-dir, it is loaded
   instead of running discovery (useful for tuning without re-running the LLM).

Usage
-----
    # Fully automatic — discovers pages, writes all RST files
    python generate_docs.py --source-dir path/to/project --output-dir docs/

    # Use a saved page manifest instead of re-discovering
    python generate_docs.py --pages-file rst_pages.json --source-dir path/to/project

    # Generate one page only (manifest must exist or --discover runs first)
    python generate_docs.py --page Examples --source-dir path/to/project

    # Print to stdout without writing files
    python generate_docs.py --dry-run --source-dir path/to/project

    # Only run discovery and save the manifest, don't generate RST yet
    python generate_docs.py --discover-only --source-dir path/to/project
"""
import logging
import ast
import argparse
import json
import re
import sys
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional

BLACKLIST_DIRS  = {"__pycache__", ".git", ".venv", "venv", ".mypy_cache", "tests", "test"}
BLACKLIST_FILES = {"setup.py", "conf.py", "helper.py"}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model & endpoint config
# ---------------------------------------------------------------------------

ENDPOINT = "http://localhost:1234/v1/chat/completions"

MODELS = {
    "orchestrator": "google/gemma-4-26b-a4b-qat",
    "analyst":      "google/gemma-4-26b-a4b-qat",
    "writer":       "openai/gpt-oss-20b",
    "reviewer":     "liquid/lfm2-24b-a2b",
    "small":        "gemma-4-e4b-it",   # used for discovery — cheap & fast
}

# ---------------------------------------------------------------------------
# DOC_STRUCTURE_PROMPT  — fully project-agnostic
# ---------------------------------------------------------------------------

DOC_STRUCTURE_PROMPT = """\
You are a senior technical writer. You have been given a compact summary of a Python
project extracted via static analysis (no LLM was used for this summary).

Your task: propose a complete, project-appropriate Sphinx documentation structure.

Rules:
- Tailor the pages to THIS project. Do not invent pages for features that do not exist.
- Between 4 and 12 pages total (include index.rst only if genuinely needed).
- Every page must map to real code in the source files listed.
- Descriptions must be specific to what the code actually does.

For each page return a JSON object with exactly these keys:
  "page_title"   : short human-readable title used as the RST heading (e.g. "Getting Started")
  "filename"     : snake_case filename WITHOUT .rst extension (e.g. "getting_started")
  "description"  : one paragraph describing what this page should cover, written
                   for a documentation writer who will expand it. Be specific —
                   mention actual class names, method names, parameters.
  "sources"      : list of .py filenames most relevant to this page (bare filenames only)

Return a JSON ARRAY — nothing else. No markdown fences. No preamble.

PROJECT SUMMARY:
{project_summary}
"""

# ---------------------------------------------------------------------------
# LLM call primitive
# ---------------------------------------------------------------------------

def _call_llm(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    endpoint: str = ENDPOINT,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }
    try:
        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        text = choice.get("content") or choice.get("reasoning_content", "")
        # Strip <think>…</think> blocks emitted by reasoning models
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    except Exception as exc:
        raise RuntimeError(f"LLM call failed [{model}]: {exc}") from exc


def _parse_json_response(raw: str):
    """Strip markdown fences and parse JSON (object or array)."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",        "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to grab the outermost JSON structure
        for pattern in (r"\[.*\]", r"\{.*\}"):
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
    return None


# ---------------------------------------------------------------------------
# Phase 0 — AST-based project scanner (no LLM)
# ---------------------------------------------------------------------------

def _ast_module_summary(source: str, filename: str, max_chars: int = 300) -> Dict:
    """
    Extract a compact, structured summary from one Python file via AST.
    Returns a dict: {filename, docstring, classes, functions, constants}.
    No LLM is used here — this is pure static analysis.
    """
    summary = {
        "filename":  filename,
        "docstring": "",
        "classes":   [],   # [{name, docstring, methods: [str]}]
        "functions": [],   # [{name, docstring}]
        "constants": [],   # [str]  — top-level UPPER_CASE names
    }
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return summary

    # Module docstring
    if (tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)):
        summary["docstring"] = str(tree.body[0].value.value)[:max_chars]

    for node in tree.body:
        # Classes
        if isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node) or ""
            methods = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not item.name.startswith("_")
            ]
            summary["classes"].append({
                "name":      node.name,
                "docstring": class_doc[:max_chars],
                "methods":   methods,
            })

        # Top-level functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                fn_doc = ast.get_docstring(node) or ""
                summary["functions"].append({
                    "name":      node.name,
                    "docstring": fn_doc[:max_chars],
                })

        # Top-level constants (UPPER_CASE assignments)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    summary["constants"].append(target.id)

    return summary


def _build_project_summary(source_base: Path) -> Dict[str, Any]:
    """
    Scan all relevant .py files in source_base and return a structured project
    summary built entirely from AST analysis — no LLM calls.

    Returns:
        {
          "files": { "filename.py": <ast_module_summary>, ... },
          "all_filenames": ["filename.py", ...]
        }
    """
    files: Dict[str, Dict] = {}

    for path in sorted(source_base.rglob("*.py")):
        if any(part in BLACKLIST_DIRS for part in path.parts):
            continue
        if path.name in BLACKLIST_FILES:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            summary = _ast_module_summary(source, path.name)
            files[path.name] = summary
        except Exception as e:
            logger.warning(f"Could not scan {path}: {e}")

    return {
        "files":         files,
        "all_filenames": sorted(files.keys()),
    }


def _format_project_summary_for_prompt(project_summary: Dict) -> str:
    """
    Render the structured project summary as compact readable text
    for inclusion in DOC_STRUCTURE_PROMPT.
    """
    lines = []
    for fname, s in project_summary["files"].items():
        lines.append(f"## {fname}")
        if s["docstring"]:
            lines.append(f"  docstring: {s['docstring'][:200]}")
        for cls in s["classes"]:
            methods_str = ", ".join(cls["methods"][:15])
            lines.append(f"  class {cls['name']}: {cls['docstring'][:120]}")
            if methods_str:
                lines.append(f"    public methods: {methods_str}")
        for fn in s["functions"]:
            lines.append(f"  def {fn['name']}(): {fn['docstring'][:120]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1 — Page discovery (one LLM call)
# ---------------------------------------------------------------------------

def discover_rst_pages(
    source_base: Path,
    model: str = MODELS["small"],
    pages_file: Optional[Path] = None,
) -> Dict[str, Dict]:
    """
    Return a pages manifest: { page_key: {description, sources, existing_rst} }

    Priority:
    1. Load from pages_file if it exists and is valid JSON.
    2. Otherwise run AST scan + one LLM call.

    The returned dict is in the same format used throughout the pipeline.
    """
    # --- Priority 1: load saved manifest ---
    if pages_file and pages_file.exists():
        logger.info(f"Loading page manifest from: {pages_file}")
        data = json.loads(pages_file.read_text(encoding="utf-8"))
        return _normalise_manifest(data)

    # --- Priority 2: AST scan + LLM ---
    logger.info("Running AST project scan (no LLM) ...")
    project_summary = _build_project_summary(source_base)
    logger.info(f"  Found {len(project_summary['files'])} source files.")

    summary_text = _format_project_summary_for_prompt(project_summary)
    prompt = DOC_STRUCTURE_PROMPT.format(project_summary=summary_text)

    logger.info(f"Discovering page structure via LLM ({model}) ...")
    raw = _call_llm(
        model,
        system="You are a technical writer. Output only valid JSON.",
        user=prompt,
        temperature=0.2,
        max_tokens=2000,
    )
    pages_list = _parse_json_response(raw)
    if not isinstance(pages_list, list):
        raise ValueError(f"Discovery LLM did not return a JSON array. Raw:\n{raw[:500]}")

    manifest = _normalise_manifest(pages_list)
    logger.info(f"Discovered {len(manifest)} page(s): {list(manifest.keys())}")
    return manifest


def _normalise_manifest(raw) -> Dict[str, Dict]:
    """
    Accept either:
      - A list  (LLM output): [{page_title, filename, description, sources}, ...]
      - A dict  (saved file): {page_key: {description, sources, existing_rst}, ...}

    Always return: {page_key: {description, sources, existing_rst}}
    """
    if isinstance(raw, dict):
        # Already in internal format — just ensure existing_rst is set
        out = {}
        for key, cfg in raw.items():
            cfg.setdefault("existing_rst", key + ".rst")
            out[key] = cfg
        return out

    if isinstance(raw, list):
        out = {}
        for item in raw:
            key = item.get("filename") or re.sub(r"\s+", "_", item.get("page_title", "page"))
            out[key] = {
                "page_title":   item.get("page_title", key),
                "description":  item.get("description", ""),
                "sources":      item.get("sources", []),
                "existing_rst": key + ".rst",
            }
        return out

    raise TypeError(f"Cannot normalise manifest of type {type(raw)}")


def save_manifest(manifest: Dict[str, Dict], path: Path):
    """Persist the page manifest to a JSON file for reuse."""
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Page manifest saved to: {path}")


# ---------------------------------------------------------------------------
# Source code parsing — AST splitter (used by analyst)
# ---------------------------------------------------------------------------

def split_script(script: str) -> List[Dict]:
    """
    Parse a Python script into structured parts for the analyst agent.
    Each method/function is capped at CHUNK chars to respect context windows.

    Returns list of:
      {"type": "header",   "name": None, "code": str}
      {"type": "class",    "name": str,  "code": str, "methods": [{name, code}]}
      {"type": "function", "name": str,  "code": str}
    """
    CHUNK        = 1500
    BLACKLIST_FN = {"verbose", "wget", "logger", "download", "tqdm"}

    try:
        tree = ast.parse(script)
    except SyntaxError:
        return [{"type": "header", "name": None, "code": script[:3000]}]

    lines = script.splitlines()

    def src(node) -> str:
        return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    def cap(text: str) -> str:
        return text[:CHUNK] + (" [truncated]" if len(text) > CHUNK else "")

    # Header: everything before the first class / function
    first_def = next(
        (n.lineno - 1 for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))),
        len(lines),
    )
    parts = [{"type": "header", "name": None, "code": "\n".join(lines[:first_def])}]

    # Classes — include class header + all method bodies
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [
            {"name": item.name, "code": cap(src(item))}
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        parts.append({
            "type":    "class",
            "name":    node.name,
            "code":    cap(src(node)),
            "methods": methods,
        })

    # Top-level functions (skip blacklisted names)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(w in node.name.lower() for w in BLACKLIST_FN):
            continue
        parts.append({"type": "function", "name": node.name, "code": cap(src(node))})

    return parts


# ---------------------------------------------------------------------------
# Source loading helpers
# ---------------------------------------------------------------------------

def _load_sources(base: Path, whitelist: Optional[List[str]] = None) -> Dict[str, List]:
    """
    Load and parse .py files under base.
    whitelist accepts bare filenames ("LLMlight.py") or stems ("LLMlight").
    Returns { "filename.py": [parts...] }
    """
    wl: Optional[set] = None
    if whitelist:
        wl = set()
        for w in whitelist:
            wl.add(w.lower())
            wl.add(Path(w).stem.lower())

    out = {}
    for path in sorted(base.rglob("*.py")):
        if any(part in BLACKLIST_DIRS for part in path.parts):
            continue
        if path.name in BLACKLIST_FILES:
            continue
        if wl is not None and path.name.lower() not in wl and path.stem.lower() not in wl:
            continue
        try:
            out[path.name] = split_script(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            out[path.name] = [{"type": "header", "name": None, "code": f"# ERROR: {e}"}]

    return out


def _load_raw_sources(base: Path, filenames: List[str]) -> Dict[str, str]:
    """Load raw text for the given filenames (used by reviewer)."""
    out = {}
    for fname in filenames:
        path = base / fname
        if not path.exists():
            found = list(base.rglob(fname))
            path = found[0] if found else None
        if path and path.exists():
            out[fname] = path.read_text(encoding="utf-8", errors="ignore")
    return out


def _load_existing_rst(name: str, base: Path) -> str:
    """Load an existing RST file if it exists, otherwise return empty string."""
    path = base / name
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _find_existing_rst(page_key: str, output_base: Path) -> str:
    """
    Auto-detect an existing RST file for page_key in output_base.
    Tries: exact match, then case-insensitive stem match.
    """
    for candidate in output_base.glob("*.rst"):
        if candidate.stem.lower() == page_key.lower():
            return candidate.read_text(encoding="utf-8", errors="ignore")
    return ""


# ---------------------------------------------------------------------------
# Analyst helpers
# ---------------------------------------------------------------------------

def _summarize_part(p: Dict, max_method_chars: int = 800) -> str:
    """Convert one parsed part to a text block for the analyst prompt."""
    if p["type"] == "header":
        return "MODULE HEADER:\n" + p["code"][:2000]
    if p["type"] == "function":
        return f"FUNCTION `{p['name']}`:\n{p['code']}"
    if p["type"] == "class":
        lines = [f"CLASS `{p['name']}`:", p["code"][:600], "", "  Methods:"]
        for m in p.get("methods", []):
            lines.append(f"  --- {m['name']} ---")
            lines.append(m["code"][:max_method_chars])
        return "\n".join(lines)
    return ""


# ---------------------------------------------------------------------------
# Agent 1 – Analyst
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = """\
You are a senior Python developer and technical writer.
Analyse source code and extract a STRUCTURED SPECIFICATION in JSON for a documentation writer.

Output ONLY valid JSON — no markdown fences, no preamble, no explanation.

JSON schema:
{
  "page_title": "str",
  "sections": [
    {
      "title": "str",
      "purpose": "str",
      "key_classes_or_functions": ["str"],
      "key_params": [{"name": "str", "type": "str", "default": "str", "description": "str"}],
      "notes": "str",
      "code_example": "str"
    }
  ],
  "important_notes": ["str"],
  "backend_notes": "str"
}
"""


def _analyze_single_file(fname: str, file_parts: List[Dict], model: str) -> str:
    """Summarize one file for the analyst map step. Returns raw text."""
    block = "\n\n".join(_summarize_part(p) for p in file_parts)[:6000]
    prompt = (
        f"FILE: {fname}\n\nCONTENT:\n{block}\n\n"
        "Summarize what this file does. List key public classes and functions "
        "with one-line descriptions. Note important parameters and return values. "
        "Keep it compact. Return ONLY JSON with keys: file, purpose, classes, functions."
    )
    return _call_llm(model, ANALYST_SYSTEM, prompt, temperature=0.2, max_tokens=1000)


def analyst_agent(
    page_name: str,
    page_description: str,
    parts: Dict[str, List],
    model: str = MODELS["analyst"],
) -> Dict:
    """MAP each file → file summary. REDUCE summaries → structured page spec."""
    file_summaries: Dict[str, str] = {}
    for fname, file_parts in parts.items():
        logger.info(f"  Analyst mapping: {fname}")
        file_summaries[fname] = _analyze_single_file(fname, file_parts, model)

    summaries_block = "\n\n".join(
        f"FILE: {k}\nSUMMARY:\n{v}" for k, v in file_summaries.items()
    )[:8000]

    reduce_prompt = (
        f"PAGE: {page_name}\n"
        f"DESCRIPTION: {page_description}\n\n"
        f"FILE SUMMARIES:\n{summaries_block}\n\n"
        "Combine the file-level insights. Produce a final documentation structure "
        "for this page. Output ONLY valid JSON matching the schema."
    )
    raw = _call_llm(model, ANALYST_SYSTEM, reduce_prompt, temperature=0.2, max_tokens=2000)
    result = _parse_json_response(raw)
    if isinstance(result, dict):
        return result
    return {"page_title": page_name, "sections": [], "raw_analyst_output": raw}


# ---------------------------------------------------------------------------
# Agent 2 – Writer
# (No library-specific facts hardcoded — relies entirely on the analyst spec)
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """\
You are an expert technical writer producing Sphinx reStructuredText (RST) documentation.

RULES:
- Use correct RST syntax: page title underlined with #, sections with =, subsections with -.
- Code examples go inside  .. code-block:: python  directives.
- Parameter tables use description-list style or  .. list-table::.
- Do NOT include .. toctree:: or page-level metadata.
- End every page with exactly this line:  .. include:: add_bottom.add
- Professional prose — concise and accurate, not marketing copy.
- All code examples must be self-contained and match the actual API described in the spec.
- Do NOT invent method names, parameter names, or behaviours not present in the spec.

Output ONLY the raw RST text. No preamble, no markdown fences.
"""


def writer_agent(
    page_name: str,
    spec: Dict,
    existing_rst: str = "",
    model: str = MODELS["writer"],
) -> str:
    spec_json = json.dumps(spec, indent=2)[:5000]
    style_hint = (
        f"\nStyle reference (do NOT copy — write fresh):\n{existing_rst[:1500]}"
        if existing_rst else ""
    )
    user = (
        f"Write the Sphinx RST documentation page for: {page_name}\n\n"
        f"SPECIFICATION (JSON):\n{spec_json}\n"
        f"{style_hint}\n\n"
        "Output the complete RST document. Start with the page title underlined with #.\n"
    )
    return _call_llm(model, WRITER_SYSTEM, user, temperature=0.4, max_tokens=3000)


# ---------------------------------------------------------------------------
# Agent 3 – Reviewer
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM = """\
You are a meticulous technical reviewer checking Sphinx RST documentation against Python source.

Your job:
1. Verify every code example against the source code.
2. Check all parameter names, types, and defaults.
3. Identify missing important public API functionality.
4. Flag RST syntax errors (wrong directive syntax, broken heading underlines).

Output JSON only:
{
  "approved": true|false,
  "issues": ["issue description", ...],
  "revised_rst": "full corrected RST string OR null if approved"
}

Minor style issues → approved=true, revised_rst=null.
Significant errors → approved=false, provide full corrected RST.
Output ONLY valid JSON — no markdown fences, no preamble.
"""


def reviewer_agent(
    page_name: str,
    draft_rst: str,
    raw_sources: Dict[str, str],
    model: str = MODELS["reviewer"],
) -> Dict:
    source_snippet = "\n\n".join(
        f"### {fname}\n{code[:2000]}" for fname, code in raw_sources.items()
    )
    user = (
        f"Page: {page_name}\n\n"
        f"CURRENT DOCUMENT:\n{draft_rst[:4000]}\n\n"
        f"SOURCE CODE (reference):\n{source_snippet}\n\n"
        "Review the document. Output JSON only.\n"
    )
    raw = _call_llm(model, REVIEWER_SYSTEM, user, temperature=0.1, max_tokens=3000)
    result = _parse_json_response(raw)
    if isinstance(result, dict):
        return result
    return {"approved": True, "issues": [], "revised_rst": None}


# ---------------------------------------------------------------------------
# Agent 4 – Orchestrator
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """\
You are the orchestration agent for an RST documentation pipeline.

Given an existing documentation page and reviewer feedback, update the page.

RULES:
- Preserve all correct content — do not rewrite for the sake of it.
- Add missing documentation where needed.
- Fix inaccuracies and broken examples.
- Keep valid Sphinx RST syntax.
- End with  .. include:: add_bottom.add
- Output ONLY the raw RST document.
"""


def orchestrator_merge(
    page_name: str,
    draft_rst: str,
    review: Dict,
    model: str = MODELS["orchestrator"],
) -> str:
    if review.get("revised_rst"):
        draft_rst = review["revised_rst"]
    issues = review.get("issues", [])
    if not issues:
        return draft_rst
    issues_text = "\n".join(f"- {i}" for i in issues)
    user = (
        f"Page: {page_name}\n\n"
        f"CURRENT RST DRAFT:\n{draft_rst[:5000]}\n\n"
        f"REVIEWER ISSUES TO FIX:\n{issues_text}\n\n"
        "Output the fully corrected RST document.\n"
    )
    return _call_llm(model, ORCHESTRATOR_SYSTEM, user, temperature=0.2, max_tokens=3000)


# ---------------------------------------------------------------------------
# Pipeline: one RST page
# ---------------------------------------------------------------------------

def generate_rst_page(
    page_name: str,
    config: Dict,
    source_base: Path,
    output_base: Path,
    max_iterations: int = 2,
    verbose: bool = True,
) -> str:
    """Run the full multi-agent pipeline for one RST page."""

    def log(msg: str):
        if verbose:
            print(f"[{page_name}] {msg}", flush=True)

    source_files = config.get("sources", [])

    # Load parsed sources for analyst
    sources_parsed = _load_sources(source_base, whitelist=source_files if source_files else None)
    if not sources_parsed:
        log("WARNING: no source files matched whitelist — loading all .py files")
        sources_parsed = _load_sources(source_base)
    log(f"Loaded {len(sources_parsed)} source file(s): {list(sources_parsed.keys())}")

    # Load raw sources for reviewer
    raw_sources = _load_raw_sources(source_base, source_files) if source_files else {}

    # Load existing RST for style reference (auto-detect by page key)
    existing_rst = _find_existing_rst(page_name, output_base)
    if not existing_rst and config.get("existing_rst"):
        existing_rst = _load_existing_rst(config["existing_rst"], output_base)

    # Step 1: Analyst
    log(f"Analyst ({MODELS['analyst']}) extracting spec ...")
    spec = analyst_agent(page_name, config["description"], sources_parsed)
    log(f"Spec sections: {[s.get('title', '?') for s in spec.get('sections', [])]}")

    # Step 2: Load pre-existing generated page OR call Writer
    existing_output = output_base / f"{page_name}.rst"
    if existing_output.exists():
        log(f"Loading existing generated page: {existing_output}")
        draft = existing_output.read_text(encoding="utf-8", errors="ignore")
    else:
        log(f"Writer ({MODELS['writer']}) drafting RST ...")
        draft = writer_agent(page_name, spec, existing_rst)
    log(f"Draft length: {len(draft)} chars")

    # Step 3+4: Review loop
    final = draft
    for iteration in range(max_iterations):
        log(f"Reviewer ({MODELS['reviewer']}) — iteration {iteration + 1} ...")
        review = reviewer_agent(page_name, final, raw_sources)

        issues   = review.get("issues", [])
        approved = review.get("approved", True)

        if issues:
            log(f"Issues ({len(issues)}):")
            for i, issue in enumerate(issues[:5], 1):
                log(f"  {i}. {issue}")
        else:
            log("No issues found.")

        if approved and not issues:
            log("Approved.")
            break

        log(f"Orchestrator ({MODELS['orchestrator']}) merging feedback ...")
        final = orchestrator_merge(page_name, final, review)
        log(f"Revised: {len(final)} chars")

    return final


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def main(
    page=None,
    source_dir=".",
    output_dir=".",
    max_iterations=2,
    dry_run=False,
    discover_only=False,
    pages_file=None,
    endpoint=None,
    orchestrator_model=None,
    analyst_model=None,
    writer_model=None,
    reviewer_model=None,
):
    global ENDPOINT

    if orchestrator_model: MODELS["orchestrator"] = orchestrator_model
    if analyst_model:      MODELS["analyst"]      = analyst_model
    if writer_model:       MODELS["writer"]        = writer_model
    if reviewer_model:     MODELS["reviewer"]      = reviewer_model
    if endpoint:           ENDPOINT = endpoint

    source_base = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve pages_file: explicit arg → auto-detect in source_dir
    pf = Path(pages_file) if pages_file else source_base / "rst_pages.json"
    if not pf.exists():
        pf = None

    # Discover page manifest
    manifest = discover_rst_pages(source_base, model=MODELS["small"], pages_file=pf)

    # Drop __init__ from the manifest — it has no useful standalone doc content;
    # its metadata is used by generate_index_rst() directly instead.
    manifest = {k: v for k, v in manifest.items()
                if not k.lower().startswith("__init__")}

    # Save manifest for reuse (always, unless it came from a pre-existing file)
    if pf is None:
        save_manifest(manifest, source_base / "rst_pages.json")

    if discover_only:
        print("\nDiscovered pages:")
        for key, cfg in manifest.items():
            print(f"  {key}")
            print(f"    description : {cfg['description'][:80]}...")
            print(f"    sources     : {cfg['sources']}")
        print(f"\nManifest saved to: {source_base / 'rst_pages.json'}")
        # Still generate the index so the user can inspect the toctree structure
        print(f"\n{'='*60}")
        print("   Generating index.rst (discover-only preview)")
        print(f"{'='*60}")
        generate_index_rst(output_path, source_base, manifest, dry_run=True)
        return manifest

    # Filter to a single page if requested
    if page:
        if page not in manifest:
            print(
                f"ERROR: page '{page}' not found in manifest. "
                f"Available: {list(manifest.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        pages_to_generate = {page: manifest[page]}
    else:
        pages_to_generate = manifest

    # ── Generate all content pages ──────────────────────────────────────────
    results: Dict[str, Optional[str]] = {}
    for page_name, config in pages_to_generate.items():
        print(f"\n{'='*60}")
        print(f"   Generating: {Path(page_name).stem}.rst")
        print(f"{'='*60}")
        try:
            rst_text = generate_rst_page(
                page_name=page_name,
                config=config,
                source_base=source_base,
                output_base=output_path,
                max_iterations=max_iterations,
                verbose=True,
            )
            results[page_name] = rst_text
            page_name = Path(page_name).stem  # remove existing extension if present

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

    # ── Generate index.rst last, after all pages exist on disk ─────────────
    # Skip when only a single page was requested (index may be incomplete).
    if page is None:
        print(f"\n{'='*60}")
        print("   Generating: index.rst")
        print(f"{'='*60}")
        try:
            index_rst = generate_index_rst(
                output_base=output_path,
                source_base=source_base,
                manifest=manifest,
                dry_run=dry_run,
            )
            results["index"] = index_rst
            if not dry_run:
                print(f"   Written: {output_path / 'index.rst'}  ({len(index_rst)} chars)")
        except Exception as exc:
            print(f"\n   ERROR generating index.rst: {exc}", file=sys.stderr)
            results["index"] = None

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("   Pipeline complete")
    print(f"{'='*60}")
    for name, rst in results.items():
        status = f"{len(rst)} chars" if rst else "FAILED"
        print(f"   {name:<30} {status}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic multi-agent Sphinx RST generator"
    )
    parser.add_argument(
        "--source-dir", default=".",
        help="Directory containing project source files (default: cwd)",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory to write .rst files (default: cwd)",
    )
    parser.add_argument(
        "--page", default=None,
        help="Generate a single named page (must appear in the manifest)",
    )
    parser.add_argument(
        "--pages-file", default=None,
        help="Path to a rst_pages.json manifest (skips LLM discovery if provided)",
    )
    parser.add_argument(
        "--discover-only", action="store_true",
        help="Run discovery and save rst_pages.json, then exit without generating RST",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=2,
        help="Max review-loop iterations per page (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated RST to stdout instead of writing files",
    )
    parser.add_argument(
        "--endpoint", default=ENDPOINT,
        help=f"LLM API endpoint (default: {ENDPOINT})",
    )
    parser.add_argument("--orchestrator-model", default=MODELS["orchestrator"])
    parser.add_argument("--analyst-model",      default=MODELS["analyst"])
    parser.add_argument("--writer-model",       default=MODELS["writer"])
    parser.add_argument("--reviewer-model",     default=MODELS["reviewer"])

    args = parser.parse_args()
    main(**vars(args))


# ---------------------------------------------------------------------------
# Index generation  (runs after all pages are written)
# ---------------------------------------------------------------------------

# Pages that should never appear in the toctree
_INDEX_SKIP = {"index", "__init__"}

# Map filename stems to human-readable captions.
# Used as fallback when a page is NOT in the manifest.
_CAPTION_FALLBACK: Dict[str, str] = {
    "summary":          "Background",
    "installation":     "Installation",
    "algorithm":        "Functions",
    "examples":         "Examples",
    "saving_and_loading": "Saving and Loading",
    "documentation":    "Documentation",
}


def _extract_project_meta(source_base: Path) -> Dict[str, str]:
    """
    Extract project metadata from conf.py and/or __init__.py via AST.
    Falls back to sensible defaults when nothing is found.

    Returns keys: name, version, author, github_user, github_repo,
                  pypi_name, description, docs_url
    """
    meta: Dict[str, str] = {
        "name":        "",
        "version":     "",
        "author":      "",
        "github_user": "",
        "github_repo": "",
        "pypi_name":   "",
        "description": "",
        "docs_url":    "",
    }

    def _read_assignments(path: Path) -> Dict[str, str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {}
        out: Dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                        out[t.id] = str(node.value.value)
        return out

    # conf.py has the cleanest project/author info
    conf = source_base / "conf.py"
    if conf.exists():
        a = _read_assignments(conf)
        meta["name"]    = a.get("project", "")
        meta["author"]  = a.get("author",  "")
        meta["version"] = a.get("version", "")

    # __init__.py for version / author fallback
    init = source_base / "__init__.py"
    if init.exists():
        a = _read_assignments(init)
        if not meta["name"]:
            # Try to infer from __doc__ first line
            doc = a.get("__doc__", "")
            first_line = doc.strip().splitlines()[0].strip() if doc.strip() else ""
            if first_line:
                meta["name"] = first_line
        meta["version"] = meta["version"] or a.get("__version__", "")
        meta["author"]  = meta["author"]  or a.get("__author__",  "")

    # Derive convenience fields from name
    name = meta["name"] or "project"
    meta["pypi_name"]   = meta["pypi_name"]   or name
    meta["github_repo"] = meta["github_repo"] or name
    meta["github_user"] = meta["github_user"] or meta["author"].split()[0] if meta["author"] else "author"
    meta["docs_url"]    = (
        meta["docs_url"]
        or f"https://erdogant.github.io/{name}/"   # keep pattern consistent with existing
    )

    return meta


def _page_caption(stem: str, manifest: Dict[str, Dict]) -> str:
    """
    Derive the toctree caption for a page stem.
    Priority: manifest page_title → _CAPTION_FALLBACK → title-cased stem.
    """
    # manifest keys are stems (e.g. "Saving_and_Loading")
    for key, cfg in manifest.items():
        if key.lower() == stem.lower():
            return cfg.get("page_title", "") or cfg.get("description", "")[:40] or key
    return _CAPTION_FALLBACK.get(stem.lower(), stem.replace("_", " ").title())


def _build_toctree_content(output_base: Path, manifest: Dict[str, Dict]) -> str:
    """
    Scan output_base for .rst files, skip index + __init__ files,
    and build one toctree block per page (each with its own :caption:).

    Returns the full RST toctree block string.
    """
    rst_files = sorted(output_base.glob("*.rst"))

    # Collect stems to include, preserving a logical order:
    # manifest order first, then any extra files found on disk.
    manifest_stems = list(manifest.keys())  # discovery order

    disk_stems = []
    for f in rst_files:
        stem = f.stem
        if stem.lower() in _INDEX_SKIP:
            continue
        if stem.lower().startswith("__init__"):
            continue
        disk_stems.append(stem)

    # Merge: manifest order → then any extra disk files not in manifest
    seen: set = set()
    ordered: List[str] = []
    for s in manifest_stems + disk_stems:
        key = s.lower()
        if key not in seen and s in disk_stems:
            seen.add(key)
            ordered.append(s)

    blocks: List[str] = []
    for stem in ordered:
        caption = _page_caption(stem, manifest)
        # RST toctree entries use the bare filename stem (no .rst)
        entry = stem  # Sphinx resolves by stem
        block = (
            f".. toctree::\n"
            f"   :maxdepth: 1\n"
            f"   :caption: {caption}\n"
            f"\n"
            f"   {entry}\n"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def generate_index_rst(
    output_base: Path,
    source_base: Path,
    manifest: Dict[str, Dict],
    dry_run: bool = False,
) -> str:
    """
    Generate index.rst by:
    1. Extracting project metadata (name, badges, description) from source.
    2. Scanning output_base for .rst files to build a dynamic toctree.
    3. Appending standard indices + badge definitions footer.

    The result is written to output_base/index.rst unless dry_run=True.
    """
    meta = _extract_project_meta(source_base)
    name = meta["name"] or "Project"
    gu   = meta["github_user"]
    gr   = meta["github_repo"]
    pn   = meta["pypi_name"]
    docs = meta["docs_url"]

    toctree_content = _build_toctree_content(output_base, manifest)

    # ── Static header (badges + intro) ─────────────────────────────────────
    header = f"""\
{name}'s documentation!
{"=" * (len(name) + len("'s documentation!"))}

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |DOI| |repo-size|

-----------------------------------

*{name}* — {meta.get("description") or f"Python package by {meta['author']}."}

.. code-block:: console

   pip install {pn}

-----------------------------------


Content
=======

{toctree_content}

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

"""

    # ── Badge definitions footer ───────────────────────────────────────────
    footer = f"""\
.. |python| image:: https://img.shields.io/pypi/pyversions/{pn}.svg
    :alt: Python
    :target: {docs}

.. |pypi| image:: https://img.shields.io/pypi/v/{pn}.svg
    :alt: PyPI version
    :target: https://pypi.org/project/{pn}/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: {docs}

.. |stars| image:: https://img.shields.io/github/stars/{gu}/{gr}
    :alt: Stars
    :target: https://github.com/{gu}/{gr}

.. |LOC| image:: https://sloc.xyz/github/{gu}/{gr}/?category=code
    :alt: lines of code
    :target: https://github.com/{gu}/{gr}

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/{pn}?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/{pn}

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/{pn}?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/{pn}

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/{gu}/{gr}/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/{gu}/{gr}.svg
    :alt: Github Forks
    :target: https://github.com/{gu}/{gr}/network

.. |open issues| image:: https://img.shields.io/github/issues/{gu}/{gr}.svg
    :alt: Open Issues
    :target: https://github.com/{gu}/{gr}/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |DOI| image:: https://zenodo.org/badge/246504758.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/246504758

.. |repo-size| image:: https://img.shields.io/github/repo-size/{gu}/{gr}
    :alt: repo-size
    :target: https://github.com/{gu}/{gr}

.. include:: add_bottom.add
"""

    index_rst = header + footer

    if dry_run:
        print("\n--- index.rst ---\n")
        print(index_rst)
    else:
        out = output_base / "index.rst"
        out.write_text(index_rst, encoding="utf-8")
        logger.info(f"Written: {out}  ({len(index_rst)} chars)")

    return index_rst