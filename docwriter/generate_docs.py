"""
generate_docs.py
================
Generic multi-agent pipeline: reads any Python project → writes Sphinx .rst files.

Architecture
------------
                    ┌─────────────┐
                    │ Orchestrator│
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────────┐
   │  Analyst  │   │  Writer   │   │   Reviewer    │
   │(code→spec)│   │(spec→rst) │   │(rst→feedback) │
   └───────────┘   └───────────┘   └───────────────┘

Guaranteed pages (always created, git+docstring enriched):
  Installation.rst  Summary.rst  Algorithm.rst  Examples.rst

Page discovery:
  1. AST scan (no LLM) → compact project summary
  2. One small-model LLM call → JSON manifest of pages
  3. Merge mandatory pages into manifest (enrich if already discovered)
  4. Optional override: rst_pages.json skips LLM discovery

Usage:
  python generate_docs.py --source-dir path/to/project --output-dir docs/
  python generate_docs.py --discover-only --source-dir path/to/project
  python generate_docs.py --page Examples --source-dir path/to/project
  python generate_docs.py --dry-run
"""

import ast
import argparse
import configparser
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLACKLIST_DIRS  = {"__pycache__", ".git", ".venv", "venv", ".mypy_cache", "tests", "test", ".secrets", ".secret", "depreciated", "old", "magweg"}
BLACKLIST_FILES = {"setup.py", "conf.py", "helper.py", ".key", ".secret", "key", "secret"}
BLACKLIST_FN = {"verbose", "wget", "logger", "download", "tqdm", "set_logger", "get_logger", "old", "messages", "print"}

# Catch numpy docstring sections
NUMPY_SECTIONS = ["Parameters", "Returns", "Examples", "Notes", "Attributes", "Raises", "See Also", "References"]

# These four rst pages are always generated, even if discovery misses them.
MANDATORY_PAGES = ["Installation", "Summary", "Algorithm", "Examples"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model & endpoint config
# ---------------------------------------------------------------------------

ENDPOINT = "http://localhost:1234/v1/chat/completions"

MODELS: Dict[str, str] = {
    "orchestrator": "openai/gpt-oss-20b",
    "analyst":      "qwen3.5-9b-glm5.1-distill-v1", # qwen3.5-9b-glm5.1-distill-v1, zai-org/glm-4.6v-flash
    "writer":       "google/gemma-4-26b-a4b-qat", # google/gemma-4-26b-a4b-qat
    "reviewer":     "liquid/lfm2-24b-a2b",
    "small":        "gemma-4-e4b-it-qat",
}

# ---------------------------------------------------------------------------
# DOC_STRUCTURE_PROMPT
# ---------------------------------------------------------------------------

DOC_STRUCTURE_PROMPT = """\
You are a senior technical writer. You have a compact static-analysis summary of a Python
project. Propose a complete, project-appropriate Sphinx documentation structure.

Rules:
- Tailor pages to THIS project. Do not invent pages for absent features.
- 4–12 pages total. The mandatory pages (Installation, Summary, Algorithm, Examples)
  will be added automatically — do NOT include them in your output.
- Every page must map to real code in the listed source files.
- Descriptions must mention actual class/method names found in the summary.

For each additional page return a JSON object:
  "page_title"  : short human-readable title (RST heading)
  "filename"    : snake_case without .rst
  "description" : one paragraph, specific to what the code actually does
  "sources"     : list of relevant .py filenames

Return a JSON ARRAY — nothing else. No markdown fences. No preamble.

PROJECT SUMMARY:
{project_summary}
"""

# ---------------------------------------------------------------------------
# LLM primitive
# ---------------------------------------------------------------------------

def _call_llm(model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream": False,
    }
    try:
        resp = requests.post(
            ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload, timeout=320,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        text = choice.get("content") or choice.get("reasoning_content", "")
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    except Exception as exc:
        raise RuntimeError(f"LLM call failed [{model}]: {exc}") from exc


def _parse_json_response(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",        "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        for pat in (r"\[.*\]", r"\{.*\}"):
            m = re.search(pat, raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
    return None


# ===========================================================================
# PHASE 0 — Git metadata extraction
# ===========================================================================

def extract_git_meta(source_base: Path) -> Dict[str, str]:
    """
    Read .git/config (and HEAD, COMMIT_EDITMSG) to extract:
      github_user, github_repo, remote_url, default_branch,
      pypi_name (= repo name), docs_url, first_commit_year, last_tag

    Returns an empty dict (no KeyError) when .git is absent.
    Falls back gracefully on any parse error.
    """
    git_dir = source_base / ".git"
    meta: Dict[str, str] = {
        "github_user":    "",
        "github_repo":    "",
        "remote_url":     "",
        "default_branch": "main",
        "pypi_name":      "",
        "docs_url":       "",
        "first_commit_year": "",
        "last_tag":       "",
        "is_git_repo":    "false",
    }

    if not git_dir.is_dir():
        logger.info("No .git directory found — skipping git metadata extraction.")
        return meta

    meta["is_git_repo"] = "true"

    # --- .git/config ---
    config_path = git_dir / "config"
    if config_path.exists():
        try:
            cfg = configparser.ConfigParser()
            cfg.read_string(config_path.read_text(encoding="utf-8", errors="ignore"))

            # Remote URL
            for section in cfg.sections():
                if "remote" in section.lower():
                    url = cfg.get(section, "url", fallback="")
                    if url:
                        meta["remote_url"] = url
                        # Parse GitHub user/repo from HTTPS or SSH URLs
                        m = re.search(
                            r"github\.com[:/]([^/]+)/([^/\.]+?)(?:\.git)?$", url
                        )
                        if m:
                            meta["github_user"] = m.group(1)
                            meta["github_repo"] = m.group(2)
                        break

            # Default branch (first [branch "..."] section)
            branch_sections = [s for s in cfg.sections() if s.startswith("branch")]
            if branch_sections:
                branch_name = re.search(r'"(.+)"', branch_sections[0])
                if branch_name:
                    meta["default_branch"] = branch_name.group(1)

        except Exception as e:
            logger.warning(f"Could not parse .git/config: {e}")

    # --- .git/packed-refs or refs/tags for last tag ---
    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        try:
            tags = re.findall(r"refs/tags/([^\s]+)", packed_refs.read_text(encoding="utf-8"))
            if tags:
                meta["last_tag"] = tags[-1]
        except Exception:
            pass

    # Also try refs/tags directory
    tags_dir = git_dir / "refs" / "tags"
    if tags_dir.is_dir():
        tag_files = sorted(tags_dir.glob("*"))
        if tag_files:
            meta["last_tag"] = meta["last_tag"] or tag_files[-1].name

    # --- Derive convenience fields ---
    repo = meta["github_repo"]
    user = meta["github_user"]
    if repo:
        meta["pypi_name"] = repo
        meta["docs_url"]  = f"https://{user}.github.io/{repo}/"

    logger.info(
        f"Git meta: user={meta['github_user']}, repo={meta['github_repo']}, "
        f"branch={meta['default_branch']}, tag={meta['last_tag']}"
    )
    return meta


# ===========================================================================
# PHASE 0 — NumPy docstring extraction
# ===========================================================================

def _parse_numpy_sections(docstring: str) -> Dict[str, str]:
    """Split a numpy-style docstring into its named sections."""
    if not docstring:
        return {}
    result: Dict[str, List[str]] = {}
    current: Optional[str] = None
    lines = docstring.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        # A numpy section header is a word (or two) followed by a dashed underline
        if (stripped in NUMPY_SECTIONS
                and i + 1 < len(lines)
                and re.match(r"^-{2,}\s*$", lines[i + 1].strip())):
            current = stripped
            result[current] = []
        elif current and re.match(r"^-{2,}\s*$", stripped):
            pass   # skip the underline itself
        elif current:
            result[current].append(line)

    return {k: "\n".join(v).strip() for k, v in result.items()}


def extract_numpy_docstrings(source_base: Path) -> Dict[str, Dict]:
    """
    Walk all .py files under source_base, extract every docstring,
    parse numpy sections, and return:

    {
      "ClassName.method_name": {
          "docstring": str,
          "Parameters": str,
          "Returns":    str,
          "Examples":   str,
          "Notes":      str,
          ...
      },
      ...
    }

    Also extracts the raw module __doc__ string from __init__.py
    under the key "__init__.__doc__".
    """
    collected: Dict[str, Dict] = {}

    for path in sorted(source_base.rglob("*.py")):
        if any(part in BLACKLIST_DIRS for part in path.parts):
            continue
        if path.name in BLACKLIST_FILES:
            continue

        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        # Module-level __doc__ assignment (common pattern: __doc__ = """...""")
        if path.name == "__init__.py":
            for node in tree.body:
                if (isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "__doc__"
                                for t in node.targets)
                        and isinstance(node.value, ast.Constant)):
                    collected["__init__.__doc__"] = {
                        "docstring": str(node.value.value),
                        **_parse_numpy_sections(str(node.value.value)),
                    }

        # Module-level docstring
        if (tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)):
            module_doc = str(tree.body[0].value.value)
            key = f"{path.stem}.__module__"
            collected[key] = {
                "docstring": module_doc,
                **_parse_numpy_sections(module_doc),
            }

        # Walk all class/function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = ast.get_docstring(node) or ""
                key = node.name
                collected[key] = {
                    "docstring": class_doc,
                    **_parse_numpy_sections(class_doc),
                }
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn_doc = ast.get_docstring(item) or ""
                        fn_key = f"{node.name}.{item.name}"
                        collected[fn_key] = {
                            "docstring": fn_doc,
                            **_parse_numpy_sections(fn_doc),
                        }
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_doc = ast.get_docstring(node) or ""
                if fn_doc:
                    collected[node.name] = {
                        "docstring": fn_doc,
                        **_parse_numpy_sections(fn_doc),
                    }

    logger.info(f"Extracted docstrings from {len(collected)} symbols.")
    return collected


def _collect_examples_from_docstrings(docstrings: Dict[str, Dict]) -> List[str]:
    """Pull all 'Examples' sections from extracted docstrings, deduplicated."""
    seen: set = set()
    examples: List[str] = []
    for info in docstrings.values():
        ex = info.get("Examples", "")
        if ex and ex not in seen:
            seen.add(ex)
            examples.append(ex)
    return examples


def _collect_pipeline_examples(source_base: Path,
                                docstrings: Dict[str, Dict]) -> List[str]:
    """
    Find the most representative pipeline examples by:
    1. Scanning __init__.__doc__ Example blocks (already structured).
    2. Collecting all docstring Examples sections.
    3. Scanning any file named examples*.py or test*.py for code blocks.
    4. Ranking by API call frequency (most-called methods appear first).

    Returns a list of example code strings, most representative first.
    """
    raw_examples: List[str] = []

    # 1. __init__.__doc__ Example blocks
    init_doc = docstrings.get("__init__.__doc__", {}).get("docstring", "")
    if init_doc:
        blocks = re.findall(
            r"Example\s*\n\s*-{2,}\s*\n(.*?)(?=\nExample\s*\n\s*-{2,}|\Z)",
            init_doc, re.DOTALL,
        )
        raw_examples.extend(b.strip() for b in blocks if b.strip())

    # 2. All docstring Examples sections
    raw_examples.extend(_collect_examples_from_docstrings(docstrings))

    # 3. Scan examples*.py files
    for path in sorted(source_base.rglob("*.py")):
        if any(part in BLACKLIST_DIRS for part in path.parts):
            continue
        if re.search(r"example|demo|tutorial|usage", path.stem, re.I):
            try:
                src = path.read_text(encoding="utf-8", errors="ignore")
                # Extract contiguous non-comment, non-blank blocks
                blocks = re.split(r"\n{3,}", src)
                for b in blocks:
                    stripped = b.strip()
                    if len(stripped) > 80:
                        raw_examples.append(stripped)
            except Exception:
                pass

    # 4. Rank: count API method calls across all examples, pick top patterns
    api_counter: Counter = Counter()
    for ex in raw_examples:
        calls = re.findall(r"\.\s*(\w+)\s*\(", ex)
        api_counter.update(calls)

    def _score(example: str) -> int:
        calls = re.findall(r"\.\s*(\w+)\s*\(", example)
        return sum(api_counter.get(c, 0) for c in calls)

    # Deduplicate then sort by score
    seen: set = set()
    unique: List[str] = []
    for ex in raw_examples:
        key = re.sub(r"\s+", " ", ex)[:200]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    ranked = sorted(unique, key=_score, reverse=True)
    return ranked[:8]   # cap at 8 most representative examples


# ===========================================================================
# PHASE 0 — AST project scanner (feeds discovery LLM)
# ===========================================================================

def _ast_module_summary(source: str, filename: str, max_chars: int = 300) -> Dict:
    summary = {
        "filename":  filename,
        "docstring": "",
        "classes":   [],
        "functions": [],
        "constants": [],
    }
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return summary

    if (tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)):
        summary["docstring"] = str(tree.body[0].value.value)[:max_chars]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not item.name.startswith("_")
            ]
            summary["classes"].append({
                "name":      node.name,
                "docstring": (ast.get_docstring(node) or "")[:max_chars],
                "methods":   methods,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                summary["functions"].append({
                    "name":      node.name,
                    "docstring": (ast.get_docstring(node) or "")[:max_chars],
                })
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    summary["constants"].append(t.id)

    return summary


def _build_project_summary(source_base: Path) -> Dict[str, Any]:
    files: Dict[str, Dict] = {}
    for path in sorted(source_base.rglob("*.py")):
        if any(part in BLACKLIST_DIRS for part in path.parts):
            continue
        if path.name in BLACKLIST_FILES:
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            files[path.name] = _ast_module_summary(src, path.name)
        except Exception as e:
            logger.warning(f"Could not scan {path}: {e}")
    return {"files": files, "all_filenames": sorted(files.keys())}


def _format_project_summary_for_prompt(ps: Dict) -> str:
    lines = []
    for fname, s in ps["files"].items():
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


# ===========================================================================
# PHASE 1 — Page discovery + mandatory page injection
# ===========================================================================

def discover_rst_pages(source_base: Path,
                       model: str = MODELS["small"],
                       pages_file: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Return manifest: { page_key: {page_title, description, sources, existing_rst} }
    Priority: pages_file → LLM discovery.
    Mandatory pages are injected/enriched afterwards in enrich_manifest().
    """
    if pages_file and pages_file.exists():
        logger.info(f"Loading manifest from: {pages_file}")
        data = json.loads(pages_file.read_text(encoding="utf-8"))
        return _normalise_manifest(data)

    logger.info("Running AST project scan ...")
    ps = _build_project_summary(source_base)
    logger.info(f"{len(ps['files'])} source files found.")

    summary_text = _format_project_summary_for_prompt(ps)
    prompt = DOC_STRUCTURE_PROMPT.format(project_summary=summary_text)

    logger.info(f"Discovering additional pages via LLM ({model}) ...")
    # run LLM
    raw = _call_llm(model, "You are a technical writer. Output only valid JSON.", prompt, temperature=0.2, max_tokens=2000)
    pages_list = _parse_json_response(raw)
    if not isinstance(pages_list, list):
        logger.warning("Discovery returned non-list — using empty extra pages.")
        pages_list = []

    manifest = _normalise_manifest(pages_list)
    logger.info(f"Discovered {len(manifest)} extra page(s): {list(manifest.keys())}")
    return manifest


def enrich_manifest(
    manifest: Dict[str, Dict],
    git_meta: Dict[str, str],
    docstrings: Dict[str, Dict],
    pipeline_examples: List[str],
    source_base: Path,
) -> Dict[str, Dict]:
    """
    Ensure the four mandatory pages exist in the manifest with rich descriptions
    built from real project data (git meta + numpy docstrings + examples).
    Existing entries are enriched, not overwritten.
    """

    # Helpers to get the best available source files
    def _find_sources(*candidates: str) -> List[str]:
        found = []
        for c in candidates:
            for path in source_base.rglob(c):
                if not any(p in BLACKLIST_DIRS for p in path.parts):
                    found.append(path.name)
        return list(dict.fromkeys(found))  # deduplicated, order preserved

    all_py = [p.name for p in sorted(source_base.rglob("*.py"))
              if not any(pt in BLACKLIST_DIRS for pt in p.parts)
              and p.name not in BLACKLIST_FILES]

    # ── Installation ────────────────────────────────────────────────────────
    name    = git_meta.get("github_repo") or git_meta.get("pypi_name") or "package"
    user    = git_meta.get("github_user", "")
    branch  = git_meta.get("default_branch", "main")
    gh_url  = git_meta.get("remote_url", "")
    tag     = git_meta.get("last_tag", "")
    # is_git  = git_meta.get("is_git_repo") == "true"

    install_desc = (
        f"Installation page for '{name}'. "
        f"GitHub user name: {user}. "
        f"{'GitHub repo: ' + gh_url + '. ' if gh_url else ''}"
        f"{'Default branch: ' + branch + '. ' if branch else ''}"
        f"{'Latest tag/release: ' + tag + '. ' if tag else ''}"
        f"Cover: conda env creation, pip install (pip install {name}), "
        f"{'GitHub install (pip install git+{gh_url}), ' if gh_url else ''}"
        f"optional dependency table extracted from imports, and uninstall steps."
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    # Collect one-sentence descriptions from class docstrings
    class_summaries = []
    for key, info in docstrings.items():
        if "." not in key and info.get("docstring"):
            first_line = info["docstring"].strip().splitlines()[0].strip()
            if first_line:
                class_summaries.append(f"{key}: {first_line[:120]}")

    summary_desc = (
        "High-level background page. Describe what the library does, its goals, "
        "and a schematic overview of the end-to-end workflow. "
        "Synthesise from: all already-written .rst files in the output directory, "
        "plus the following class descriptions extracted from source docstrings:\n"
        + "\n".join(class_summaries[:10])
    )

    # ── Algorithm and Core functions ───────────────────────────────────────────────────────────
    # Collect Parameters sections — they describe algorithm inputs well
    param_snippets = []
    for key, info in docstrings.items():
        params = info.get("Parameters", "")
        notes  = info.get("Notes", "")
        if params:
            param_snippets.append(f"[{key}] Parameters:\n{params[:2000]}")
        if notes:
            param_snippets.append(f"[{key}] Notes:\n{notes[:2000]}")

    algo_desc = (
        "Technical workflow page. Describe the step-by-step algorithm and core functions: "
        "data ingestion, chunking, embedding, storage, retrieval, and inference. "
        "Include a schematic ASCII workflow diagram showing how components connect. "
        "Draw on these parameter/notes sections from the source docstrings:\n"
        + "\n".join(param_snippets[:8])
    )

    # ── Examples ────────────────────────────────────────────────────────────
    # Embed the top pipeline examples directly in the description so the writer
    # has concrete code to render — no hallucination needed.
    top_examples = pipeline_examples
    examples_desc = (
        "Practical examples page. Show the most common user workflows as runnable "
        "code blocks. Use the following examples extracted from docstrings and "
        "example files (ranked by API usage frequency):\n\n"
        + "\n\n---\n\n".join(top_examples)
    )

    mandatory: Dict[str, Dict] = {
        "Installation": {
            "page_title":   "Installation",
            "description":  install_desc,
            "sources":      _find_sources("__init__.py", "setup.py", "requirements*.txt"),
            "existing_rst": "Installation.rst",
            # Pass git_meta directly so the writer template can use it
            "_git_meta":    git_meta,
        },
        "Summary": {
            "page_title":  "Summary",
            "description": summary_desc,
            "sources":     all_py[:6],
            "existing_rst": "Summary.rst",
            "_class_summaries": class_summaries,
        },
        "Algorithm": {
            "page_title":  "Algorithm",
            "description": algo_desc,
            "sources":     all_py[:8],
            "existing_rst": "Algorithm.rst",
            "_param_snippets": param_snippets[:8],
        },
        "Examples": {
            "page_title":  "Examples",
            "description": examples_desc,
            "sources":     _find_sources("__init__.py", "example*.py", "demo*.py"),
            "existing_rst": "Examples.rst",
            "_pipeline_examples": top_examples,
        },
    }

    # Merge: mandatory first (so they appear at top), then discovered extras.
    # If a mandatory page was already discovered, enrich its description but
    # keep any sources the LLM added.
    out: Dict[str, Dict] = {}
    for key, mand_cfg in mandatory.items():
        if key in manifest:
            existing = manifest[key]
            mand_cfg["description"] = mand_cfg["description"]
            mand_cfg["sources"] = list(dict.fromkeys(mand_cfg["sources"] + existing.get("sources", [])))
        # Store
        out[key] = mand_cfg

    for key, cfg in manifest.items():
        norm_key = key.lower().replace(" ", "_")
        if not any(norm_key == m.lower() for m in mandatory):
            out[key] = cfg
    # Return
    return out


def _normalise_manifest(raw) -> Dict[str, Dict]:
    if isinstance(raw, dict):
        out = {}
        for key, cfg in raw.items():
            cfg.setdefault("existing_rst", key + ".rst")
            cfg.setdefault("page_title", key)
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
    # Strip private keys (_git_meta etc.) before saving
    clean = {}
    for k, v in manifest.items():
        clean[k] = {sk: sv for sk, sv in v.items() if not sk.startswith("_")}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    logger.info(f"Manifest saved: {path}")


# ===========================================================================
# Source loading helpers
# ===========================================================================

def extract_functions(script: str) -> List[Dict]:
    # Maximum code block chunk is 2500 chars
    CHUNK = 2500
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return [{"type": "header", "name": None, "code": script[:CHUNK]}]

    lines = script.splitlines()

    def src(n) -> str:
        return "\n".join(lines[n.lineno - 1: n.end_lineno])

    def truncate(t: str) -> str:
        return t[:CHUNK] + (" [truncated]" if len(t) > CHUNK else "")

    first_def = next((n.lineno - 1 for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), len(lines))
    parts = [{"type": "header", "name": None, "code": "\n".join(lines[:first_def])}]

    for node in tree.body:

        # ---------------- CLASS ----------------
        if isinstance(node, ast.ClassDef):

            class_code = truncate(src(node))

            # method extraction + description per method
            methods = []

            for m in tqdm(node.body):
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(b in m.name.lower() for b in BLACKLIST_FN):
                        continue
                    # Show message
                    logger.info(f'[Code Agent]> Describing function: {m.name}')
                    # Get code
                    m_code = truncate(src(m))
                    # Retrieve the description using LLM based on the code
                    description = describe(m.name, m_code, context=f"Method inside class {node.name}")
                    # Append
                    methods.append({
                        "name": m.name,
                        "code": m_code,
                        "description": description,
                    })

            parts.append({
                "type": "class",
                "name": node.name,
                "code": class_code,
                "description": describe(node.name, class_code, context="Python class"),
                "methods": methods
            })

        # ---------------- FUNCTION ----------------
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            if any(b in node.name.lower() for b in BLACKLIST_FN):
                continue

            # Get function code
            func_code = truncate(src(node))
            # Make the description of the code
            description = describe(node.name, func_code, context="Standalone function")

            parts.append({
                "type": "function",
                "name": node.name,
                "code": func_code,
                "description": description,
            })

    return parts

def describe(name, code, context=""):
    # Build prompt
    prompt = build_description_prompt(name=name, code=code, context=context)
    # Run LLM
    out = _call_llm(MODELS["analyst"], TECHNICAL_WRITER_SYSTEM, prompt)
    # Return
    return out

def _load_sources(base: Path, whitelist: Optional[List[str]] = None) -> Dict[str, List]:
    wl: Optional[set] = None
    if whitelist:
        wl = {w.lower() for w in whitelist} | {Path(w).stem.lower() for w in whitelist}
    out = {}
    for path in sorted(base.rglob("*.py")):
        if any(part in BLACKLIST_DIRS for part in path.parts):
            continue
        if path.name in BLACKLIST_FILES:
            continue
        if wl is not None and path.name.lower() not in wl and path.stem.lower() not in wl:
            continue
        try:
            # Get code seperated on class and functions
            out[path.name] = extract_functions(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            out[path.name] = [{"type": "header", "name": None, "code": f"# ERROR: {e}"}]
    return out


def _load_raw_sources(base: Path, filenames: List[str]) -> Dict[str, str]:
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
    path = base / name
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _find_existing_rst(page_key: str, output_base: Path) -> str:
    """Find and return the content of an existing RST file recursively."""
    page_key = page_key.lower()

    rst_file = next(
        (
            f
            for f in sorted(output_base.rglob("*.rst"))
            if f.stem.lower() == page_key
        ),
        None,
    )

    return (rst_file.read_text(encoding="utf-8", errors="ignore") if rst_file else "")

# ===========================================================================
# Analyst helpers
# ===========================================================================

def _summarize_part(p: Dict, chunk: int = 3000) -> str:
    if p["type"] == "header":
        return "MODULE HEADER:\n" + p["code"][:chunk]

    # Retrieve for func
    if p["type"] == "function":
        content = p.get('description', '') + '\n\nCODE:\n\n' + p.get('code', '')
        content = content[:chunk]
        # Return
        return f"FUNCTION `{p['name']}`:\n{content}"

    # Retrieve for class
    if p["type"] == "class":
        class_content =  p.get('description', '') + '\n\nCODE:\n\n' + p.get('code', '')
        class_content = class_content[:chunk]

        # Get lines
        lines = [f"CLASS `{p['name']}`:", class_content, "", "  Methods:"]
        # Go through all func in the class
        for m in p.get("methods", []):
            func_content = p.get('description', '') + '\n\nCODE:\n\n' + p.get('code', '')
            func_content = func_content[:chunk]
            # Make lines
            lines += [f"  --- {m['name']} ---", func_content]
        # Return
        return "\n".join(lines)

    return ""

# ===========================================================================
# Agents
# ===========================================================================

ANALYST_SYSTEM = """\
You are a senior Python developer and technical writer.
Analyse source code and extract a STRUCTURED SPECIFICATION in JSON for a documentation writer.
Output ONLY valid JSON — no markdown fences, no preamble.

JSON schema:
{
  "page_title": "str",
  "sections": [
    {
      "title": "str",
      "purpose": "str",
      "key_classes_or_functions": ["str"],
      "key_params": [{"name":"str","type":"str","default":"str","description":"str"}],
      "notes": "str",
      "code_example": "str"
    }
  ],
  "important_notes": ["str"],
  "backend_notes": "str"
}
"""

TECHNICAL_WRITER_SYSTEM = """
You are a senior software engineer and code analyst.

Your job is to explain Python functions clearly, precisely, and technically.

Rules:
- Do not guess behavior not supported by code
- Focus on input/output behavior and logic
- Be concise but complete
- Prefer deterministic descriptions over vague explanations
- Explain what the role of this function is in the complete pipeline
"""

def _analyze_single_file(fname: str, file_parts: List[Dict]) -> str:
    block = "\n\n".join(_summarize_part(p) for p in file_parts)[:10000]
    prompt = (
        f"FILE: {fname}\n\nCONTENT:\n{block}\n\n"
        "Summarize this file. List key public classes and functions with one-line "
        "descriptions and important parameters. Return ONLY JSON: "
        "{file, purpose, classes, functions}"
    )
    # Call llm
    out = _call_llm(MODELS["writer"], ANALYST_SYSTEM, prompt, temperature=0.2, max_tokens=1000)
    # return
    return out


def analyst_agent(page_name: str, page_description: str, parts: Dict[str, List], extra_context: str = "") -> Dict:
    """MAP each file → summary. REDUCE → structured page spec."""
    logger.info(f"[Product Owner Agent]> {page_name} - ({MODELS['writer']}) working on specifications ...")
    file_summaries: Dict[str, str] = {}

    # Go through all files and functions
    for fname, file_parts in parts.items():
        logger.info(f"[Analyst Agent]> Working on all functions of {fname}")
        file_summaries[fname] = _analyze_single_file(fname, file_parts)

    # Create summary block
    summaries_block = "\n\n".join(f"FILE: {k}\nSUMMARY:\n{v}" for k, v in file_summaries.items())[:20000]

    # Prompt
    reduce_prompt = (
        f"PAGE: {page_name}\n"
        f"DESCRIPTION: {page_description[:2000]}\n\n"
        f"FILE SUMMARIES:\n{summaries_block}\n"
        + (f"\nADDITIONAL CONTEXT:\n{extra_context[:1500]}\n" if extra_context else "")
        + "\nCombine insights. Produce the page spec JSON.")

    # Run LLM
    logger.info(f"[Analyst Agent]> Collected all information and is now going to combine and structure all information.")
    raw = _call_llm(MODELS["writer"], ANALYST_SYSTEM, reduce_prompt, temperature=0.2, max_tokens=2000)
    # structure
    result = _parse_json_response(raw)
    # Create final string
    out = result if isinstance(result, dict) else {"page_title": page_name, "sections": [], "raw": raw}
    # Return
    return out


WRITER_SYSTEM = """\
You are an expert technical writer producing Sphinx reStructuredText (RST) documentation.

RULES:
- RST syntax: title underlined with #, sections with =, subsections with -.
- Code examples: .. code-block:: python
- Parameter tables: description-list or .. list-table::
- No .. toctree:: or page-level metadata.
- End every page with:  .. include:: add_bottom.add
- Accurate — do NOT invent API names not in the spec.
- For Algorithm pages: include the core functions and an ASCII diagram of the workflow inside
  a .. code-block:: text  directive.
- For Installation pages: include separate code-block:: console blocks for
  conda, pip, and github install commands.

Output ONLY raw RST text. No preamble, no markdown fences.
"""


def writer_agent(page_name: str, spec: Dict, existing_rst: str = "") -> str:
    logger.info(f"[Writer Agent]> {page_name} - ({MODELS['writer']}) drafting RST ...")
    # dump json
    spec_json = json.dumps(spec, indent=2)[:5000]
    style_hint = (
        f"\nStyle reference (do NOT copy — write fresh):\n{existing_rst[:1500]}"
        if existing_rst else ""
    )
    user = (
        f"Write the Sphinx RST page: {page_name}\n\n"
        f"SPECIFICATION:\n{spec_json}\n"
        f"{style_hint}\n\n"
        "Start with page title underlined by #. Output complete RST.\n"
    )
    # Call llm
    raw = _call_llm(MODELS["writer"], WRITER_SYSTEM, user, temperature=0.4, max_tokens=3000)
    # Return
    return raw


REVIEWER_SYSTEM = """\
You are a technical reviewer checking Sphinx RST against Python source code.

Check:
1. Code examples match the actual API.
2. Parameter names/types/defaults are correct.
3. No missing important public API.
4. Valid RST syntax.

Output JSON only:
{"approved": bool, "issues": ["str",...], "revised_rst": "str or null"}
Minor issues → approved=true. Significant errors → approved=false + revised_rst.
"""


def reviewer_agent(page_name: str, draft_rst: str, raw_sources: Dict[str, str]) -> Dict:
    snippet = "\n\n".join(f"### {f}\n{c[:2000]}" for f, c in raw_sources.items())
    user = (
        f"Page: {page_name}\n\nDRAFT:\n{draft_rst[:4000]}\n\n"
        f"SOURCE:\n{snippet}\n\nReview. Output JSON only.\n"
    )
    # run LLM
    raw = _call_llm(MODELS["reviewer"], REVIEWER_SYSTEM, user, temperature=0.1, max_tokens=3000)
    result = _parse_json_response(raw)
    return result if isinstance(result, dict) else {"approved": True, "issues": [], "revised_rst": None}


ORCHESTRATOR_SYSTEM = """\
You are an orchestration agent merging reviewer feedback into an RST draft.
Preserve correct content. Fix errors. Keep valid RST. End with .. include:: add_bottom.add
Output ONLY the corrected RST document.
"""


def orchestrator_merge(page_name: str, draft_rst: str, review: Dict) -> str:
    logger.info("[Orchestrator Agent]> merging feedback of {page_name}...")

    if review.get("revised_rst"):
        draft_rst = review["revised_rst"]
    issues = review.get("issues", [])
    if not issues:
        return draft_rst
    user = (
        f"Page: {page_name}\n\nDRAFT:\n{draft_rst[:5000]}\n\n"
        f"ISSUES:\n" + "\n".join(f"- {i}" for i in issues) +
        "\n\nOutput the fully corrected RST.\n"
    )
    # run LLM
    raw = _call_llm(MODELS["orchestrator"], ORCHESTRATOR_SYSTEM, user, temperature=0.2, max_tokens=3000)
    # return
    return raw


# ===========================================================================
# Special page: Installation (template-based, minimal LLM)
# ===========================================================================

def _build_installation_rst(config: Dict, source_base: Path) -> str:
    """
    Build Installation.rst mostly from template + git meta, with one small
    LLM call only for the optional-dependencies paragraph.
    """
    git  = config.get("_git_meta", {})
    name = git.get("github_repo") or git.get("pypi_name") or "package"
    user = git.get("github_user", "")
    url  = git.get("remote_url", "")
    branch = git.get("default_branch", "main")
    tag    = git.get("last_tag", "")
    docs   = git.get("docs_url", "")

    gh_install = f"pip install git+{url}" if url else f"pip install git+https://github.com/{user}/{name}"

    # Detect optional deps from import statements in source files
    optional_imports: set = set()
    optional_keywords = {
        "sentence_transformers": "sentence-transformers",
        "hnswlib": "hnswlib",
        "memvid": "memvid",
        "distfit": "distfit",
        "llama_cpp": "llama-cpp-python",
        "torch": "torch",
        "transformers": "transformers",
        "faiss": "faiss-cpu",
        "sklearn": "scikit-learn",
        "pymupdf": "pymupdf",
        "fitz": "pymupdf",
        "ebooklib": "ebooklib",
        "bs4": "beautifulsoup4",
    }
    for path in source_base.rglob("*.py"):
        if any(p in BLACKLIST_DIRS for p in path.parts):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            for keyword, pkg in optional_keywords.items():
                if keyword in src:
                    optional_imports.add(pkg)
        except Exception:
            pass

    opt_table_rows = ""
    if optional_imports:
        rows = "\n".join(
            f"   * - ``pip install {pkg}``\n     - Optional feature"
            for pkg in sorted(optional_imports)
        )
        opt_table_rows = (
            "\nOptional Dependencies\n"
            "*********************\n\n"
            ".. list-table::\n"
            "   :header-rows: 1\n\n"
            "   * - Package\n"
            "     - Purpose\n"
            + rows + "\n"
        )

    tag_note = f"\n.. note::\n\n   Latest release: ``{tag}``\n" if tag else ""
    github_note = (
        f"\n.. note::\n\n   Source code: `{name} on GitHub <{url}>`_\n"
        if url else ""
    )
    docs_note = (
        f"\n.. tip::\n\n   Documentation: `{docs} <{docs}>`_\n"
        if docs else ""
    )

    rst = f"""\
Installation
############

{tag_note}{github_note}{docs_note}

Create environment
******************

.. code-block:: console

   conda create -n env_{name} python=3.10
   conda activate env_{name}


Install from PyPI
*****************

.. code-block:: console

   pip install {name}

   # Force update to the latest version
   pip install -U {name}


Install from GitHub
*******************

.. code-block:: console

   pip install git+{url if url else f"https://github.com/{user}/{name}"}

   # Install a specific branch
   pip install git+{url if url else f"https://github.com/{user}/{name}"}@{branch}


{opt_table_rows}

Uninstall
############

Remove environment
******************

.. code-block:: console

   conda env list
   conda env remove --name env_{name}
   conda env list


Remove package
**************

.. code-block:: console

   pip uninstall {name}


"""
    return rst


# ===========================================================================
# Special page: Summary (reads already-written RSTs + docstrings)
# ===========================================================================

def _build_summary_spec(config: Dict, output_base: Path,
                        docstrings: Dict[str, Dict]) -> Dict:
    """Build analyst spec for Summary by reading existing RST files."""
    written_rst_texts = []
    for rst_file in sorted(output_base.glob("*.rst")):
        if rst_file.stem.lower() in ("index", "summary"):
            continue
        text = rst_file.read_text(encoding="utf-8", errors="ignore")
        # Take only the first 500 chars of each (enough for summary context)
        written_rst_texts.append(f"## {rst_file.stem}\n{text[:500]}")

    class_summaries = config.get("_class_summaries", [])

    extra_context = (
        "ALREADY-WRITTEN RST PAGES (use these to synthesise a coherent summary):\n"
        + "\n\n".join(written_rst_texts[:8])
        + "\n\nCLASS DESCRIPTIONS FROM SOURCE:\n"
        + "\n".join(class_summaries[:10])
    )
    return {
        "page_title": "Summary",
        "extra_context": extra_context,
        "sections": [
            {"title": "Background",
             "purpose": "What the library does and why it exists",
             "notes": extra_context[:1000]},
            {"title": "Output",
             "purpose": "What users can build or produce with the library"},
            {"title": "Schematic Overview",
             "purpose": "High-level workflow diagram (ASCII art in code-block:: text)"},
        ]
    }


# ===========================================================================
# Pipeline: one RST page
# ===========================================================================
def generate_rst_page(
    page_name: str,
    config: Dict,
    source_base: Path,
    output_base: Path,
    docstrings: Dict[str, Dict],
    max_iterations: int = 2,
    verbose: bool = True,
) -> str:

    if page_name == "Installation":
        logger.info(f"Building Installation.rst from template + git metadata ...")
        return _build_installation_rst(config, source_base)

    # load page sources
    sources_parsed, raw_sources = _load_page_sources(config, source_base)
    # load page rst
    existing_rst = _load_page_rst(page_name, config, output_base)
    # Build page specifications
    spec = _build_page_spec(page_name, config, output_base, docstrings, sources_parsed)
    # Create draft
    draft = _load_or_create_draft(page_name, spec, existing_rst, output_base)
    # Review and finalize
    fin = _review_and_finalize(page_name, draft, raw_sources, max_iterations)
    # Return
    return fin

def _load_page_sources(config, source_base):
    source_files = config.get("sources", [])
    sources = _load_sources(source_base, whitelist=source_files or None)
    if not sources: 
        logger.warning(f"No source files matched — loading all .py files")
        sources = _load_sources(source_base)

    # Get raw sources
    raw_sources = _load_raw_sources(source_base, source_files) if source_files else {}

    # return
    logger.info(f"Loaded {len(sources)} file(s): {list(sources.keys())}")
    return sources, raw_sources

def _load_page_rst(page_name, config, output_base):
    # look for existing rst page
    rst = _find_existing_rst(page_name, output_base)

    if not rst and config.get("existing_rst"):
        rst = _load_existing_rst(config["existing_rst"], output_base)
    # return
    return rst

def _build_page_spec(page_name, config, output_base, docstrings, sources):
    if page_name == "Summary":
        logger.info(f"Building Summary specifications for {page_name} ...")

        spec = _build_summary_spec(config, output_base, docstrings)
        spec.pop("extra_context", None)
        return spec

    extra_context = ""

    if page_name == "Algorithm":
        extra_context = "\n".join(config.get("_param_snippets", [])[:5])

    elif page_name == "Examples":
        examples = config.get("_pipeline_examples", [])
        extra_context = (
            "MOST-USED PIPELINE EXAMPLES "
            "(render these as code-block:: python):\n\n"
            + "\n\n---\n\n".join(examples))

    # Run analyst agent
    raw = analyst_agent(page_name, config["description"], sources, extra_context=extra_context)
    # return
    return raw

def _load_or_create_draft(page_name, spec, existing_rst, output_base):
    # create output file
    output_file = output_base / f"{page_name}.rst"

    if output_file.exists():
        logger.info(f"{page_name} - Loading existing generated page: {output_file}")
        raw = output_file.read_text(encoding="utf-8", errors="ignore")
        return raw

    # Run the agent
    draft = writer_agent(page_name, spec, existing_rst)

    # return
    logger.info(f"{page_name} - Draft created: {len(draft)} chars")
    return draft

def _review_and_finalize(page_name, draft, raw_sources, max_iterations):
    final = draft
    for iteration in range(max_iterations):

        review = reviewer_agent(page_name, final, raw_sources)
        issues = review.get("issues", [])
        approved = review.get("approved", True)

        if approved and not issues:
            logger.info(f"{page_name} - Approved.")
            break

        for i, issue in enumerate(issues[:5], 1):
            logger.info(f"{page_name} - {i}. {issue}")

        # Run agent
        final = orchestrator_merge(page_name, final, review)

    return final


# ===========================================================================
# Index generation
# ===========================================================================

_INDEX_SKIP = {"index", "__init__"}

_CAPTION_FALLBACK: Dict[str, str] = {
    "summary":            "Background",
    "installation":       "Installation",
    "algorithm":          "Functions",
    "examples":           "Examples",
    "saving_and_loading": "Saving and Loading",
    "documentation":      "Documentation",
}


def _extract_project_meta(source_base: Path, git_meta: Dict[str, str]) -> Dict[str, str]:
    meta = {
        "name":    git_meta.get("github_repo", ""),
        "version": "",
        "author":  git_meta.get("github_user", ""),
        "pypi_name": git_meta.get("pypi_name", ""),
        "github_user": git_meta.get("github_user", ""),
        "github_repo": git_meta.get("github_repo", ""),
        "docs_url": git_meta.get("docs_url", ""),
        "description": "",
    }

    def _read_asgn(path: Path) -> Dict[str, str]:
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

    conf = source_base / "conf.py"
    if conf.exists():
        a = _read_asgn(conf)
        meta["name"]    = meta["name"]    or a.get("project", "")
        meta["author"]  = meta["author"]  or a.get("author",  "")
        meta["version"] = meta["version"] or a.get("version", "")

    init = source_base / "__init__.py"
    if init.exists():
        a = _read_asgn(init)
        meta["version"] = meta["version"] or a.get("__version__", "")
        meta["author"]  = meta["author"]  or a.get("__author__",  "")

    name = meta["name"] or "project"
    meta["pypi_name"]    = meta["pypi_name"]    or name
    meta["github_repo"]  = meta["github_repo"]  or name
    meta["docs_url"]     = meta["docs_url"]      or f"https://github.com/{meta['github_user']}/{name}"

    return meta


def _page_caption(stem: str, manifest: Dict[str, Dict]) -> str:
    for key, cfg in manifest.items():
        if key.lower() == stem.lower():
            return cfg.get("page_title", "") or key
    return _CAPTION_FALLBACK.get(stem.lower(), stem.replace("_", " ").title())


def _build_toctree_content(output_base: Path, manifest: Dict[str, Dict]) -> str:
    manifest_stems = list(manifest.keys())
    disk_stems = [
        f.stem for f in sorted(output_base.glob("*.rst"))
        if f.stem.lower() not in _INDEX_SKIP
        and not f.stem.lower().startswith("__init__")
    ]
    seen: set = set()
    ordered: List[str] = []
    for s in manifest_stems + disk_stems:
        if s.lower() not in seen and s in disk_stems:
            seen.add(s.lower())
            ordered.append(s)
    blocks = []
    for stem in ordered:
        caption = _page_caption(stem, manifest)
        blocks.append(
            f".. toctree::\n"
            f"   :maxdepth: 1\n"
            f"   :caption: {caption}\n\n"
            f"   {stem}\n"
        )
    return "\n\n".join(blocks)


def generate_index_rst(output_base: Path, source_base: Path,
                       manifest: Dict[str, Dict],
                       git_meta: Dict[str, str],
                       dry_run: bool = False) -> str:
    meta = _extract_project_meta(source_base, git_meta)
    name = meta["name"] or "Project"
    gu   = meta["github_user"]
    gr   = meta["github_repo"]
    pn   = meta["pypi_name"]
    docs = meta["docs_url"]

    toctree = _build_toctree_content(output_base, manifest)

    header = f"""\
{name}'s documentation!
{"=" * (len(name) + len("'s documentation!"))}

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |DOI| |repo-size|

-----------------------------------

*{name}* — Python library by {meta['author']}.

.. code-block:: console

   pip install {pn}

-----------------------------------


Content
=======

{toctree}

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

"""

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


def build_description_prompt(name: str, code: str, context: str = "") -> str:
    return f"""
FUNCTION NAME:
{name}

CONTEXT:
{context}

SOURCE CODE:
{code}

TASK:
Explain what this function does.

OUTPUT FORMAT:
1. Summary (maximum 3-4 sentences)
2. Key steps (bullet points)
3. Inputs and outputs
"""

# ===========================================================================
# Main
# ===========================================================================

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
    # Apply runtime
    _apply_runtime_config(endpoint, orchestrator_model, analyst_model, writer_model, reviewer_model)
    # Prepare paths
    source_base, output_path = _prepare_paths(source_dir, output_dir)

    # =======================
    # Phase 0 collect context
    # =======================
    git_meta, docstrings, pipeline_examples = _phase0_collect_context(source_base)

    # =======================
    # Phase 1
    # =======================
    manifest = _phase1_discover_pages(source_base, pages_file, git_meta, docstrings, pipeline_examples, discover_only, output_path, dry_run)

    if discover_only:
        return manifest

    # =======================
    # Phase 2 — per-source draft pages
    # =======================
    results = _phase2_generate_pages(manifest, page, source_base, output_path, docstrings, max_iterations, dry_run)

    # =======================
    # Phase 3 — compile drafts into dense final pages
    # =======================
    _phase3_compile(page, output_path, source_base, manifest, docstrings, git_meta, dry_run, results)

    # =======================
    # Phase 4 — index.rst
    # =======================
    _phase4_generate_index(page, output_path, source_base, manifest, git_meta, dry_run, results)

    # =======================
    # Summary
    # =======================
    _print_summary(results)

    # return
    return results

def _phase0_collect_context(source_base):
    print("\n[Phase 0] Extracting git metadata ...")
    git_meta = extract_git_meta(source_base)

    print("[Phase 0] Extracting numpy docstrings ...")
    docstrings = extract_numpy_docstrings(source_base)

    print("[Phase 0] Collecting pipeline examples ...")
    pipeline_examples = _collect_pipeline_examples(source_base, docstrings)

    print(f"  Found {len(pipeline_examples)} ranked example(s).")

    return git_meta, docstrings, pipeline_examples

def _phase1_discover_pages(source_base, pages_file, git_meta, docstrings, pipeline_examples, discover_only, output_path, dry_run):
    pf = Path(pages_file) if pages_file else source_base / "rst_pages.json"
    pf = pf if pf.exists() else None

    manifest = discover_rst_pages(source_base, model=MODELS["small"], pages_file=pf)

    manifest = {
        k: v for k, v in manifest.items()
        if not k.lower().startswith("__init__")
    }

    manifest = enrich_manifest(manifest, git_meta, docstrings, pipeline_examples, source_base)

    if pf is None:
        save_manifest(manifest, source_base / "rst_pages.json")

    if discover_only:
        print("\nDiscovered pages:")
        for k, v in manifest.items():
            print(f"  {k:<25}  {v['description'][:70]}...")
        print(f"\nManifest: {source_base / 'rst_pages.json'}")

        generate_index_rst(output_path, source_base, manifest, git_meta, dry_run=True)

    # Return
    return manifest

def _phase2_generate_pages(manifest, page, source_base, output_path, docstrings, max_iterations, dry_run):
    mandatory = set(MANDATORY_PAGES)
    extra_pages = {k: v for k, v in manifest.items() if k not in mandatory}
    mand_pages = {k: manifest[k] for k in MANDATORY_PAGES if k in manifest}

    if page:
        if page not in manifest:
            print(f"ERROR: '{page}' not in manifest.", file=sys.stderr)
            sys.exit(1)
        extra_pages, mand_pages = {page: manifest[page]}, {}

    results = {}

    def run_page(pname, pcfg):
        print(f"\n{'='*60}\nProcessing: {pname}\n{'='*60}")
        try:
            rst = generate_rst_page(page_name=pname, config=pcfg, source_base=source_base, output_base=output_path, docstrings=docstrings, max_iterations=max_iterations, verbose=True)
            results[pname] = rst

            if dry_run:
                print(f"\n--- {pname}.rst ---\n{rst}")
            else:
                out_file = output_path / f"{pname}.rst"
                out_file.write_text(rst, encoding="utf-8")
                print(f"Written: {out_file}")

        except Exception as e:
            print(f"ERROR: {pname}: {e}", file=sys.stderr)
            results[pname] = None

    for p, c in extra_pages.items():
        run_page(p, c)

    for p in MANDATORY_PAGES:
        if p in mand_pages:
            run_page(p, mand_pages[p])
    # Return
    return results

# ===========================================================================
# PHASE 3 — Compilation: route draft RSTs → dense final pages
# ===========================================================================

# Pages that are never archived (always kept as final output).
FINAL_PAGES = {"Installation", "Summary", "Algorithm", "Examples", "index"}

COMPILER_ROUTER_SYSTEM = """\
You are a documentation architect. You receive a set of draft RST documentation pages
generated from individual Python source files. Your job is to route every meaningful
content block into one of the final documentation pages.

Final pages available:
  - Installation  : setup, dependencies, environment, uninstall
  - Summary       : background, overview, goals, workflow diagram, what the library produces
  - Algorithm     : technical workflow, chunking, embedding, retrieval, core functions,
                    parameters, statistical validation, internal architecture
  - Examples      : runnable code examples, usage patterns, full pipelines

For each draft page decide ONE best-fit target from the list above.
If a draft covers a distinct topic that does not fit any of those four, propose a
new short snake_case page name (e.g. "api_reference", "configuration").

Output ONLY a JSON object — no fences, no preamble:
{
  "routing": {
    "<draft_stem>": "<target_page_name>",
    ...
  },
  "new_pages": ["page_name_if_any"]
}
"""

COMPILER_WRITER_SYSTEM = """\
You are a senior technical writer producing dense, comprehensive Sphinx RST documentation.

DENSITY REQUIREMENTS — every section must have:
  - At least 3–5 full prose paragraphs (not bullet lists of one-liners)
  - A complete parameter table (.. list-table:: with Type, Default, Description columns)
    for every class or function that has parameters
  - At least one runnable .. code-block:: python example per major function/class
  - Clear explanation of WHY each component exists, not just WHAT it does

RST RULES:
  - Page title underlined with #
  - Section headings underlined with =
  - Subsection headings underlined with -
  - Code examples: .. code-block:: python
  - Console commands: .. code-block:: console
  - Parameter tables: .. list-table:: with :header-rows: 1
  - Workflow diagrams: .. code-block:: text  (ASCII art)
  - No .. toctree::  No page-level metadata.
  - End the page with EXACTLY:  .. include:: add_bottom.add

QUALITY RULES:
  - Do NOT invent method names or parameters absent from the provided content
  - Do NOT write placeholder text like "See documentation for details"
  - Merge duplicate information — if two drafts say the same thing, write it once, better
  - Prefer concrete specifics over vague generalities

Output ONLY raw RST. No preamble. No markdown fences.
"""


def _read_all_drafts(output_base: Path, skip: set = None) -> Dict[str, str]:
    """Read all .rst files in output_base except those in skip set and index."""
    skip = (skip or set()) | {"index"}
    out: Dict[str, str] = {}
    for rst_file in sorted(output_base.glob("*.rst")):
        if rst_file.stem.lower() in {s.lower() for s in skip}:
            continue
        try:
            out[rst_file.stem] = rst_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    return out


def compiler_router_agent(drafts: Dict[str, str]) -> Dict[str, Any]:
    """
    Ask the LLM to route each draft RST stem to a final page name.
    Returns {"routing": {stem: target}, "new_pages": [str]}.
    """
    # Build a compact listing: stem + first 300 chars of content
    listing = "\n\n".join(
        f"DRAFT: {stem}\nCONTENT PREVIEW:\n{text[:400]}"
        for stem, text in drafts.items()
    )
    # Cap to avoid flooding context
    listing = listing[:12000]

    user = (
        f"Route these draft documentation pages to final pages.\n\n"
        f"{listing}\n\n"
        "Output ONLY the JSON routing object."
    )
    logger.info(f"[Compiler Router Agent]> ({MODELS['orchestrator']}) routing {len(drafts)} drafts ...")
    raw = _call_llm(MODELS["orchestrator"], COMPILER_ROUTER_SYSTEM, user, temperature=0.1, max_tokens=1000)
    result = _parse_json_response(raw)
    if not isinstance(result, dict):
        # Fallback: route everything to Algorithm
        logger.warning("Router returned invalid JSON — routing all drafts to Algorithm.")
        result = {"routing": {s: "Algorithm" for s in drafts}, "new_pages": []}
    result.setdefault("routing", {})
    result.setdefault("new_pages", [])
    return result


def compiler_writer_agent(
    page_name: str,
    content_blocks: List[str],
    raw_sources: Dict[str, str],
    git_meta: Dict[str, str],
    docstrings: Dict[str, Dict],
    existing_rst: str = "",
) -> str:
    """
    Write a dense, comprehensive RST page from routed content blocks.
    content_blocks: list of draft RST texts assigned to this page.
    raw_sources:    raw .py source snippets relevant to this page.
    """
    # Combine all routed content (cap per block to manage context)
    combined = "\n\n" + ("=" * 60) + "\n\n".join(
        f"[FROM: {i + 1}]\n{block[:3000]}"
        for i, block in enumerate(content_blocks)
    )
    combined = combined[:14000]

    # Source snippets for grounding
    source_snippet = "\n\n".join(
        f"### {fname}\n{code[:1500]}" for fname, code in list(raw_sources.items())[:4]
    )

    # Style hint from existing reference RST
    style_hint = (
        f"\nStyle reference (structure only — do NOT copy text):\n{existing_rst[:800]}"
        if existing_rst else ""
    )

    # Extra metadata for Installation
    install_hint = ""
    if page_name == "Installation":
        name   = git_meta.get("github_repo") or git_meta.get("pypi_name") or "package"
        url    = git_meta.get("remote_url", "")
        branch = git_meta.get("default_branch", "main")
        install_hint = (
            f"\nGIT METADATA (use for exact install commands):\n"
            f"  package name : {name}\n"
            f"  remote url   : {url}\n"
            f"  branch       : {branch}\n"
        )

    user = (
        f"Write the final, comprehensive Sphinx RST page for: **{page_name}**\n\n"
        f"ROUTED CONTENT (from draft pages — extract, merge, and expand):\n{combined}\n"
        f"\nSOURCE CODE REFERENCE:\n{source_snippet}\n"
        f"{install_hint}{style_hint}\n\n"
        "Requirements:\n"
        "- Every section: minimum 3 full prose paragraphs\n"
        "- Every class/function with params: include a complete .. list-table:: parameter table\n"
        "- Every major feature: include a .. code-block:: python example\n"
        "- Merge duplicate content from different drafts into one clear explanation\n"
        "- Do not write placeholder text\n\n"
        "Output the complete RST document. Start with the page title underlined by #.\n"
    )

    logger.info(f"[Compiler Writer Agent]> ({MODELS['writer']}) building: {page_name} "
                f"({len(content_blocks)} routed block(s)) ...")
    return _call_llm(MODELS["writer"], COMPILER_WRITER_SYSTEM, user,
                     temperature=0.35, max_tokens=4000)


def archive_drafts(output_base: Path, keep: set) -> List[Path]:
    """
    Move .rst files NOT in keep into output_base/_drafts/.
    Returns list of archived paths.
    """
    drafts_dir = output_base / "_drafts"
    drafts_dir.mkdir(exist_ok=True)
    archived: List[Path] = []
    for rst_file in sorted(output_base.glob("*.rst")):
        if rst_file.stem.lower() not in {k.lower() for k in keep}:
            dest = drafts_dir / rst_file.name
            rst_file.rename(dest)
            archived.append(dest)
            logger.info(f"Archived: {rst_file.name} → _drafts/")
    return archived


def compile_rst_pages(
    output_base: Path,
    source_base: Path,
    manifest: Dict[str, Dict],
    docstrings: Dict[str, Dict],
    git_meta: Dict[str, str],
    dry_run: bool = False,
) -> Dict[str, str]:
    """
    Phase 3: read all draft RSTs, route content, write dense final pages, archive drafts.

    Steps:
      3A  Read all draft RSTs from output_base
      3B  Router agent: assign each draft → target final page
      3C  For each final page: compiler_writer_agent produces dense RST
      3D  Reviewer pass (one iteration) per final page
      3E  Archive superseded drafts to _drafts/
    Returns {page_name: rst_text}
    """
    print(f"\n{'='*60}\n[Phase 3] Compiling draft pages into final documentation\n{'='*60}")

    # 3A — Read drafts (skip Installation — it's template-built and already correct)
    all_drafts = _read_all_drafts(output_base, skip={"Installation", "index"})
    if not all_drafts:
        logger.warning("No draft RST files found in output directory — skipping compilation.")
        return {}

    print(f"  Found {len(all_drafts)} draft page(s): {sorted(all_drafts.keys())}")

    # 3B — Route
    routing_result = compiler_router_agent(all_drafts)
    routing: Dict[str, str] = routing_result.get("routing", {})
    new_pages: List[str]    = routing_result.get("new_pages", [])

    # Unrouted drafts default to Algorithm
    for stem in all_drafts:
        if stem not in routing:
            routing[stem] = "Algorithm"

    print(f"  Routing: { {v: [k for k,vv in routing.items() if vv==v] for v in set(routing.values())} }")
    if new_pages:
        print(f"  New pages proposed: {new_pages}")

    # Build the set of final page names (mandatory + any new)
    final_page_names = list(FINAL_PAGES) + [p for p in new_pages if p not in FINAL_PAGES]

    # 3C — Invert routing: target → [list of draft content blocks]
    target_blocks: Dict[str, List[str]] = {p: [] for p in final_page_names}
    for draft_stem, target in routing.items():
        if target not in target_blocks:
            target_blocks[target] = []
        if draft_stem in all_drafts:
            target_blocks[target].append(all_drafts[draft_stem])

    # 3D — Write each final page (skip Installation — already written)
    compiled: Dict[str, str] = {}
    for page_name in final_page_names:
        if page_name in ("Installation", "index"):
            continue

        blocks = target_blocks.get(page_name, [])
        if not blocks:
            logger.info(f"{page_name} - No content routed — skipping.")
            continue

        # Gather raw source files relevant to this page
        page_cfg = manifest.get(page_name, {})
        page_sources = page_cfg.get("sources", [])
        raw_sources = _load_raw_sources(source_base, page_sources) if page_sources else {}

        # Style reference from original hand-written RST if available
        existing_rst = _find_existing_rst(page_name, output_base)

        # Write
        rst_text = compiler_writer_agent(
            page_name=page_name,
            content_blocks=blocks,
            raw_sources=raw_sources,
            git_meta=git_meta,
            docstrings=docstrings,
            existing_rst=existing_rst,
        )

        # One reviewer pass
        print(f"  Reviewer checking: {page_name} ...")
        review = reviewer_agent(page_name, rst_text, raw_sources)
        if not review.get("approved", True) or review.get("issues"):
            rst_text = orchestrator_merge(page_name, rst_text, review)

        compiled[page_name] = rst_text

        if not dry_run:
            out_file = output_base / f"{page_name}.rst"
            out_file.write_text(rst_text, encoding="utf-8")
            print(f"  Written: {out_file}  ({len(rst_text)} chars)")
        else:
            print(f"\n--- {page_name}.rst ---\n{rst_text[:1000]}...\n")

    # 3E — Archive superseded drafts (keep final pages + Installation + index)
    keep_stems = {p.lower() for p in FINAL_PAGES} | {p.lower() for p in compiled} | {"installation"}
    print("\n  Archiving superseded draft pages ...")
    archived = archive_drafts(output_base, keep={k for k in keep_stems})
    print(f"  Archived {len(archived)} draft file(s) to _drafts/")

    return compiled


def _phase3_compile(page, output_path, source_base, manifest,
                    docstrings, git_meta, dry_run, results):
    """Run the compilation phase (Phase 3) unless a single page was requested."""
    if page is not None:
        # Single-page mode: skip compilation
        return

    compiled = compile_rst_pages(
        output_base=output_path,
        source_base=source_base,
        manifest=manifest,
        docstrings=docstrings,
        git_meta=git_meta,
        dry_run=dry_run,
    )
    results.update(compiled)


def _phase4_generate_index(page, output_path, source_base, manifest,
                           git_meta, dry_run, results):
    """Generate index.rst as the very last step."""
    if page is not None:
        return
    print(f"\n{'='*60}\n[Phase 4] Generating: index.rst\n{'='*60}")
    try:
        results["index"] = generate_index_rst(
            output_path, source_base, manifest, git_meta, dry_run=dry_run
        )
    except Exception as e:
        print(f"ERROR: index.rst: {e}", file=sys.stderr)
        results["index"] = None


def _print_summary(results):
    print(f"\n{'='*60}\nPipeline complete\n{'='*60}")
    for name, rst in results.items():
        status = f"{len(rst)} chars" if rst else "FAILED"
        print(f"   {name:<30} {status}")

def _apply_runtime_config(endpoint, orchestrator_model, analyst_model, writer_model, reviewer_model):

    global ENDPOINT

    if orchestrator_model:
        MODELS["orchestrator"] = orchestrator_model
    if analyst_model:
        MODELS["analyst"] = analyst_model
    if writer_model:
        MODELS["writer"] = writer_model
    if reviewer_model:
        MODELS["reviewer"] = reviewer_model

    if endpoint:
        ENDPOINT = endpoint
        
def _prepare_paths(source_dir: str, output_dir: str):
    source_base = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return source_base, output_path

# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic multi-agent Sphinx RST generator")
    parser.add_argument("--source-dir",       default=".")
    parser.add_argument("--output-dir",        default=".")
    parser.add_argument("--page",              default=None,
                        help="Generate a single named page")
    parser.add_argument("--pages-file",        default=None,
                        help="Path to rst_pages.json (skips LLM discovery)")
    parser.add_argument("--discover-only",     action="store_true",
                        help="Discover and save manifest, print preview, exit")
    parser.add_argument("--max-iterations",    type=int, default=2)
    parser.add_argument("--dry-run",           action="store_true")
    parser.add_argument("--endpoint",          default=ENDPOINT)
    parser.add_argument("--orchestrator-model", default=MODELS["orchestrator"])
    parser.add_argument("--analyst-model",      default=MODELS["analyst"])
    parser.add_argument("--writer-model",       default=MODELS["writer"])
    parser.add_argument("--reviewer-model",     default=MODELS["reviewer"])
    args = parser.parse_args()
    main(**vars(args))