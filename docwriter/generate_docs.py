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

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLACKLIST_DIRS  = {"__pycache__", ".git", ".venv", "venv", ".mypy_cache", "tests", "test", ".secrets", ".secret", "depreciated", "old", "magweg"}
BLACKLIST_FILES = {"setup.py", "conf.py", "helper.py", ".key", ".secret", "key", "secret"}
BLACKLIST_FN = {"verbose", "wget", "logger", "download", "tqdm", "set_logger", "get_logger", "old"}

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
    "orchestrator": "google/gemma-4-26b-a4b-qat",
    "analyst":      "google/gemma-4-26b-a4b-qat",
    "writer":       "openai/gpt-oss-20b",
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

def _call_llm(model: str, system: str, user: str,
              temperature: float = 0.3, max_tokens: int = 2048) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }
    try:
        resp = requests.post(
            ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload, timeout=180,
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
    logger.info(f"  {len(ps['files'])} source files found.")

    summary_text = _format_project_summary_for_prompt(ps)
    prompt = DOC_STRUCTURE_PROMPT.format(project_summary=summary_text)

    logger.info(f"Discovering additional pages via LLM ({model}) ...")
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
    is_git  = git_meta.get("is_git_repo") == "true"

    install_desc = (
        f"Installation page for '{name}'. "
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
            param_snippets.append(f"[{key}] Parameters:\n{params[:300]}")
        if notes:
            param_snippets.append(f"[{key}] Notes:\n{notes[:200]}")

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
    top_examples = pipeline_examples[:4]
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
            mand_cfg["sources"]     = list(dict.fromkeys(
                mand_cfg["sources"] + existing.get("sources", [])
            ))
        out[key] = mand_cfg

    for key, cfg in manifest.items():
        norm_key = key.lower().replace(" ", "_")
        if not any(norm_key == m.lower() for m in mandatory):
            out[key] = cfg

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

def split_script(script: str) -> List[Dict]:
    CHUNK = 1500
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return [{"type": "header", "name": None, "code": script[:3000]}]
    lines = script.splitlines()
    def src(n) -> str: return "\n".join(lines[n.lineno - 1 : n.end_lineno])
    def cap(t: str) -> str: return t[:CHUNK] + (" [truncated]" if len(t) > CHUNK else "")

    first_def = next(
        (n.lineno - 1 for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))),
        len(lines),
    )
    parts = [{"type": "header", "name": None, "code": "\n".join(lines[:first_def])}]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                {"name": i.name, "code": cap(src(i))}
                for i in node.body
                if isinstance(i, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            parts.append({"type": "class", "name": node.name,
                          "code": cap(src(node)), "methods": methods})

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not any(w in node.name.lower() for w in BLACKLIST_FN):
                parts.append({"type": "function", "name": node.name, "code": cap(src(node))})

    return parts


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
            out[path.name] = split_script(path.read_text(encoding="utf-8", errors="ignore"))
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
    for candidate in output_base.glob("*.rst"):
        if candidate.stem.lower() == page_key.lower():
            return candidate.read_text(encoding="utf-8", errors="ignore")
    return ""


# ===========================================================================
# Analyst helpers
# ===========================================================================

def _summarize_part(p: Dict, max_method_chars: int = 800) -> str:
    if p["type"] == "header":
        return "MODULE HEADER:\n" + p["code"][:2000]
    if p["type"] == "function":
        return f"FUNCTION `{p['name']}`:\n{p['code']}"
    if p["type"] == "class":
        lines = [f"CLASS `{p['name']}`:", p["code"][:600], "", "  Methods:"]
        for m in p.get("methods", []):
            lines += [f"  --- {m['name']} ---", m["code"][:max_method_chars]]
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


def _analyze_single_file(fname: str, file_parts: List[Dict], model: str) -> str:
    block = "\n\n".join(_summarize_part(p) for p in file_parts)[:6000]
    prompt = (
        f"FILE: {fname}\n\nCONTENT:\n{block}\n\n"
        "Summarize this file. List key public classes and functions with one-line "
        "descriptions and important parameters. Return ONLY JSON: "
        "{file, purpose, classes, functions}"
    )
    return _call_llm(model, ANALYST_SYSTEM, prompt, temperature=0.2, max_tokens=1000)


def analyst_agent(page_name: str, page_description: str,
                  parts: Dict[str, List],
                  extra_context: str = "",
                  model: str = MODELS["analyst"]) -> Dict:
    """MAP each file → summary. REDUCE → structured page spec."""
    file_summaries: Dict[str, str] = {}
    for fname, file_parts in parts.items():
        logger.info(f"  Analyst Agent is handling: {fname}")
        file_summaries[fname] = _analyze_single_file(fname, file_parts, model)

    summaries_block = "\n\n".join(f"FILE: {k}\nSUMMARY:\n{v}" for k, v in file_summaries.items())[:7000]

    reduce_prompt = (
        f"PAGE: {page_name}\n"
        f"DESCRIPTION: {page_description[:2000]}\n\n"
        f"FILE SUMMARIES:\n{summaries_block}\n"
        + (f"\nADDITIONAL CONTEXT:\n{extra_context[:1500]}\n" if extra_context else "")
        + "\nCombine insights. Produce the page spec JSON."
    )
    raw = _call_llm(model, ANALYST_SYSTEM, reduce_prompt, temperature=0.2, max_tokens=2000)
    result = _parse_json_response(raw)
    return result if isinstance(result, dict) else {"page_title": page_name, "sections": [], "raw": raw}


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


def writer_agent(page_name: str, spec: Dict, existing_rst: str = "",
                 model: str = MODELS["writer"]) -> str:
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
    return _call_llm(model, WRITER_SYSTEM, user, temperature=0.4, max_tokens=3000)


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


def reviewer_agent(page_name: str, draft_rst: str,
                   raw_sources: Dict[str, str],
                   model: str = MODELS["reviewer"]) -> Dict:
    snippet = "\n\n".join(f"### {f}\n{c[:2000]}" for f, c in raw_sources.items())
    user = (
        f"Page: {page_name}\n\nDRAFT:\n{draft_rst[:4000]}\n\n"
        f"SOURCE:\n{snippet}\n\nReview. Output JSON only.\n"
    )
    raw = _call_llm(model, REVIEWER_SYSTEM, user, temperature=0.1, max_tokens=3000)
    result = _parse_json_response(raw)
    return result if isinstance(result, dict) else {"approved": True, "issues": [], "revised_rst": None}


ORCHESTRATOR_SYSTEM = """\
You are an orchestration agent merging reviewer feedback into an RST draft.
Preserve correct content. Fix errors. Keep valid RST. End with .. include:: add_bottom.add
Output ONLY the corrected RST document.
"""


def orchestrator_merge(page_name: str, draft_rst: str, review: Dict,
                       model: str = MODELS["orchestrator"]) -> str:
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
    return _call_llm(model, ORCHESTRATOR_SYSTEM, user, temperature=0.2, max_tokens=3000)


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
*********

Remove environment
==================

.. code-block:: console

   conda env list
   conda env remove --name env_{name}
   conda env list


Remove package
==============

.. code-block:: console

   pip uninstall {name}


.. include:: add_bottom.add
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

def generate_rst_page(page_name: str, config: Dict,
                      source_base: Path, output_base: Path,
                      docstrings: Dict[str, Dict],
                      max_iterations: int = 2,
                      verbose: bool = True) -> str:

    def log(msg: str):
        if verbose:
            print(f"[{page_name}] {msg}", flush=True)

    # ── Special case: Installation — template, no LLM analyst needed ────────
    if page_name == "Installation":
        log("Building Installation.rst from template + git metadata ...")
        return _build_installation_rst(config, source_base)

    source_files = config.get("sources", [])
    sources_parsed = _load_sources(source_base, whitelist=source_files or None)
    if not sources_parsed:
        log("WARNING: no source files matched — loading all .py files")
        sources_parsed = _load_sources(source_base)
    log(f"Loaded {len(sources_parsed)} file(s): {list(sources_parsed.keys())}")

    raw_sources = _load_raw_sources(source_base, source_files) if source_files else {}

    existing_rst = _find_existing_rst(page_name, output_base)
    if not existing_rst and config.get("existing_rst"):
        existing_rst = _load_existing_rst(config["existing_rst"], output_base)

    # ── Summary — special spec that reads already-written pages ─────────────
    if page_name == "Summary":
        log("Building Summary spec from existing RST pages + docstrings ...")
        spec = _build_summary_spec(config, output_base, docstrings)
        extra_context = spec.pop("extra_context", "")
    else:
        # ── Algorithm / Examples / generic pages ───────────────────────────
        extra_context = ""
        if page_name == "Algorithm":
            snippets = config.get("_param_snippets", [])
            extra_context = "\n".join(snippets[:5])
        elif page_name == "Examples":
            pipeline_examples = config.get("_pipeline_examples", [])
            extra_context = (
                "MOST-USED PIPELINE EXAMPLES (render these as code-block:: python):\n\n"
                + "\n\n---\n\n".join(pipeline_examples)
            )

        log(f"Analyst Agent ({MODELS['analyst']}) extracting spec ...")
        spec = analyst_agent(page_name, config["description"], sources_parsed, extra_context=extra_context)

    log(f"Spec sections: {[s.get('title','?') for s in spec.get('sections',[])]}")

    # Re-use existing generated output if it already exists
    existing_output = output_base / f"{page_name}.rst"
    if existing_output.exists():
        log(f"Loading existing generated page: {existing_output}")
        draft = existing_output.read_text(encoding="utf-8", errors="ignore")
    else:
        log(f"Writer Agent ({MODELS['writer']}) drafting RST ...")
        draft = writer_agent(page_name, spec, existing_rst)
    log(f"Draft is created: {len(draft)} chars")

    # ── Review loop ──────────────────────────────────────────────────────────
    final = draft
    for iteration in range(max_iterations):
        log(f"Reviewer Agent iteration {iteration + 1} ...")
        review = reviewer_agent(page_name, final, raw_sources)
        issues   = review.get("issues", [])
        approved = review.get("approved", True)
        if issues:
            for i, issue in enumerate(issues[:5], 1):
                log(f"  {i}. {issue}")
        if approved and not issues:
            log("Approved.")
            break
        log(f"Orchestrator Agent is merging feedback ...")
        final = orchestrator_merge(page_name, final, review)
        log(f"Revised: {len(final)} chars")

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
    global ENDPOINT
    if orchestrator_model: MODELS["orchestrator"] = orchestrator_model
    if analyst_model:      MODELS["analyst"]      = analyst_model
    if writer_model:       MODELS["writer"]        = writer_model
    if reviewer_model:     MODELS["reviewer"]      = reviewer_model
    if endpoint:           ENDPOINT = endpoint

    source_base = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Phase 0: enrich from git + docstrings (no LLM) ──────────────────────
    print("\n[Phase 0] Extracting git metadata ...")
    git_meta = extract_git_meta(source_base)

    print("[Phase 0] Extracting numpy docstrings ...")
    docstrings = extract_numpy_docstrings(source_base)

    print("[Phase 0] Collecting pipeline examples ...")
    pipeline_examples = _collect_pipeline_examples(source_base, docstrings)
    print(f"  Found {len(pipeline_examples)} ranked example(s).")

    # ── Phase 1: discover pages ──────────────────────────────────────────────
    pf = Path(pages_file) if pages_file else source_base / "rst_pages.json"
    if not pf.exists():
        pf = None

    manifest = discover_rst_pages(source_base, model=MODELS["small"], pages_file=pf)

    # Drop any __init__ pages the LLM might have added
    manifest = {k: v for k, v in manifest.items()
                if not k.lower().startswith("__init__")}

    # Inject mandatory pages with enriched descriptions
    manifest = enrich_manifest(manifest, git_meta, docstrings, pipeline_examples, source_base)

    if pf is None:
        save_manifest(manifest, source_base / "rst_pages.json")

    if discover_only:
        print("\nDiscovered pages:")
        for key, cfg in manifest.items():
            print(f"  {key:<25}  {cfg['description'][:70]}...")
        print(f"\nManifest: {source_base / 'rst_pages.json'}")
        generate_index_rst(output_path, source_base, manifest, git_meta, dry_run=True)
        return manifest

    # ── Phase 2: generate pages ──────────────────────────────────────────────
    # Order: mandatory pages last so Summary can read already-written RSTs.
    mandatory_keys = set(MANDATORY_PAGES)
    extra_pages  = {k: v for k, v in manifest.items() if k not in mandatory_keys}
    mand_pages   = {k: manifest[k] for k in MANDATORY_PAGES if k in manifest}

    # If a single page was requested, just run that one
    if page:
        if page not in manifest:
            print(f"ERROR: '{page}' not in manifest. Available: {list(manifest.keys())}", file=sys.stderr)
            sys.exit(1)
        pages_to_generate = {page: manifest[page]}
        mand_pages = {}
        extra_pages = pages_to_generate
    else:
        pages_to_generate = {**extra_pages, **mand_pages}

    results: Dict[str, Optional[str]] = {}

    def _run_page(pname: str, pcfg: Dict):
        print(f"\n{'='*60}\n   Generating: {pname}\n{'='*60}")
        try:
            rst = generate_rst_page(
                page_name=pname, config=pcfg,
                source_base=source_base, output_base=output_path,
                docstrings=docstrings,
                max_iterations=max_iterations, verbose=True,
            )
            results[pname] = rst
            if dry_run:
                print(f"\n--- {pname}.rst ---\n{rst}")
            else:
                out_file = output_path / f"{pname}.rst"
                out_file.write_text(rst, encoding="utf-8")
                print(f"\n   Written: {out_file}  ({len(rst)} chars)")
        except Exception as exc:
            print(f"\n   ERROR: {pname}: {exc}", file=sys.stderr)
            results[pname] = None

    # Extra pages first
    for pname, pcfg in extra_pages.items():
        _run_page(pname, pcfg)

    # Mandatory pages after (Summary reads other RSTs)
    for pname in MANDATORY_PAGES:
        if pname in mand_pages:
            _run_page(pname, mand_pages[pname])

    # ── Phase 3: index ───────────────────────────────────────────────────────
    if page is None:
        print(f"\n{'='*60}\n   Generating: index.rst\n{'='*60}")
        try:
            index_rst = generate_index_rst(
                output_path, source_base, manifest, git_meta, dry_run=dry_run
            )
            results["index"] = index_rst
        except Exception as exc:
            print(f"\n   ERROR: index.rst: {exc}", file=sys.stderr)
            results["index"] = None

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n   Pipeline complete\n{'='*60}")
    for n, rst in results.items():
        status = f"{len(rst)} chars" if rst else "FAILED"
        print(f"   {n:<30} {status}")

    return results


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