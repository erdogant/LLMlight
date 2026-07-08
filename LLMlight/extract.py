"""Structured information extraction via Google's LangExtract.

LangExtract (https://github.com/google/langextract) uses an LLM to pull
structured, schema-consistent data out of unstructured text, and -- unlike a
plain "return me some JSON" prompt -- grounds every extraction to its exact
character span in the source text. That grounding is what makes the
interactive visualization in this module possible: every extracted entity
can be highlighted back in its original context.

This module is a thin wrapper around the ``langextract`` package. It adds:

- ``build_example()``       -- build few-shot examples from plain dicts,
                               without importing ``langextract.data`` yourself.
- ``extract()``             -- thin pass-through to ``lx.extract()``, with
                               optional routing through an OpenAI-compatible
                               local endpoint (e.g. an LLMlight/LM
                               Studio/vLLM server) instead of a cloud model.
- ``save_results()``        -- persist result(s) to a .jsonl file.
- ``visualize()``           -- generate the interactive, self-contained HTML
                               visualization from result(s) or a .jsonl file.

``langextract`` is an optional dependency: install with ``pip install
langextract`` (add ``langextract[openai]`` if you want to point it at an
OpenAI-compatible endpoint, which is how ``LLMlight.extract()`` on the main
class talks to a local model by default).
"""
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def _require_langextract():
    """Import langextract lazily, with a helpful error if it's missing."""
    try:
        import langextract as lx
    except ImportError as exc:
        raise ImportError(
            "The 'langextract' package is required for this feature. "
            "Install it with: pip install langextract "
            "(use pip install \"langextract[openai]\" if you plan to route "
            "extraction through a local OpenAI-compatible endpoint, e.g. "
            "LM Studio/vLLM, via base_url) "
            "(see https://github.com/google/langextract)"
        ) from exc
    return lx


def build_example(text, extractions):
    """
    Build a single ``lx.data.ExampleData`` few-shot example from plain
    Python objects, so callers don't need to import ``langextract.data``.

    Parameters
    ----------
    text : str
        The example source text.
    extractions : list of dict
        Each dict describes one extraction and needs:
            - 'extraction_class' (str): the category, e.g. 'character'.
            - 'extraction_text'  (str): verbatim text from `text` (no
              paraphrasing -- LangExtract grounds this back to `text`).
            - 'attributes' (dict, optional): extra context/metadata.

    Returns
    -------
    lx.data.ExampleData

    Examples
    --------
    >>> example = build_example(
    ...     text="ROMEO. But soft! What light through yonder window breaks?",
    ...     extractions=[
    ...         {"extraction_class": "character", "extraction_text": "ROMEO",
    ...          "attributes": {"emotional_state": "wonder"}},
    ...         {"extraction_class": "emotion", "extraction_text": "But soft!",
    ...          "attributes": {"feeling": "gentle awe"}},
    ...     ],
    ... )
    """
    lx = _require_langextract()
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(
                extraction_class=e['extraction_class'],
                extraction_text=e['extraction_text'],
                attributes=e.get('attributes'),
            )
            for e in extractions
        ],
    )


def extract(text_or_documents,
            prompt_description,
            examples,
            model_id=None,
            api_key=None,
            base_url=None,
            provider=None,
            provider_kwargs=None,
            config=None,
            **kwargs):
    """
    Run LLM-based structured information extraction with precise source
    grounding (thin wrapper around ``langextract.extract``).

    Parameters
    ----------
    text_or_documents : str
        Raw text to extract from. LangExtract also accepts a URL or its own
        ``Document``/list-of-``Document`` objects -- see their docs.
    prompt_description : str
        Free-text instructions describing what to extract.
    examples : list of lx.data.ExampleData
        Few-shot examples that steer extraction. Build these with
        :func:`build_example` if you don't want to import ``langextract``
        directly.
    model_id : str, optional
        e.g. ``'gemini-3.5-flash'``, ``'gpt-4o'``, ``'gemma2:2b'`` (Ollama),
        or the name of a model served behind `base_url`. Required unless
        `config` fully specifies the model.
    api_key : str, optional
        Passed through to LangExtract. Falls back to the
        ``LANGEXTRACT_API_KEY`` / ``OPENAI_API_KEY`` environment variables
        per LangExtract's own resolution when omitted. Local/self-hosted
        endpoints usually don't need a real key.
    base_url : str, optional
        Point LangExtract at an OpenAI-compatible endpoint (e.g. a local
        LM Studio/vLLM server, such as the one an ``LLMlight`` instance
        talks to) instead of a cloud provider. Implies
        ``provider='openai'`` unless `provider` is set explicitly. Requires
        the ``openai`` package: ``pip install "langextract[openai]"``.
        When set and `api_key` is omitted, a harmless placeholder
        (``'lm-studio'``) is used, since the OpenAI SDK requires a
        non-empty key even when the local server doesn't check it.
    provider : str, optional
        Explicit LangExtract provider name (e.g. ``'openai'``). Inferred
        from `base_url` when omitted.
    provider_kwargs : dict, optional
        Extra kwargs merged into ``ModelConfig.provider_kwargs`` (e.g.
        custom headers).
    config : langextract.factory.ModelConfig, optional
        Pass a fully-built ``ModelConfig`` yourself to bypass the
        `base_url`/`provider`/`provider_kwargs` convenience path entirely.
    **kwargs :
        Forwarded to ``lx.extract`` (e.g. ``extraction_passes``,
        ``max_workers``, ``max_char_buffer``, ``fence_output``).

    Returns
    -------
    AnnotatedDocument
        LangExtract's result object. Iterate ``result.extractions``; each
        item exposes ``.extraction_class``, ``.extraction_text``,
        ``.attributes``, and ``.char_interval`` (``None`` when the model's
        output couldn't be grounded in the source text -- filter these out
        for high-precision use, e.g.
        ``[e for e in result.extractions if e.char_interval]``).

    Examples
    --------
    >>> import LLMlight.extract as lx_extract
    >>> examples = [lx_extract.build_example(
    ...     text="ROMEO. But soft! What light through yonder window breaks?",
    ...     extractions=[{"extraction_class": "character",
    ...                    "extraction_text": "ROMEO",
    ...                    "attributes": {"emotional_state": "wonder"}}],
    ... )]
    >>> result = lx_extract.extract(
    ...     text_or_documents="Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    ...     prompt_description="Extract characters, emotions, and relationships in order of appearance.",
    ...     examples=examples,
    ...     model_id="gemini-3.5-flash",
    ... )
    >>> html = lx_extract.visualize(result, output_html="visualization.html")
    """
    lx = _require_langextract()

    if config is None and base_url:
        from langextract.factory import ModelConfig
        # The OpenAI SDK requires a non-empty api_key even when talking to a
        # local server (LM Studio, vLLM, ...) that doesn't actually check
        # it, so fall back to a harmless placeholder rather than erroring.
        pk = {"base_url": base_url, "api_key": api_key or "lm-studio"}
        if provider_kwargs:
            pk.update(provider_kwargs)
        config = ModelConfig(
            model_id=model_id,
            provider=provider or "openai",
            provider_kwargs=pk,
        )

    extract_kwargs = dict(
        text_or_documents=text_or_documents,
        prompt_description=prompt_description,
        examples=examples,
        **kwargs,
    )

    if config is not None:
        extract_kwargs['config'] = config
        logger.info('Running langextract extraction via explicit ModelConfig.')
    else:
        extract_kwargs['model_id'] = model_id
        if api_key:
            extract_kwargs['api_key'] = api_key
        logger.info(f'Running langextract extraction with model_id={model_id}.')

    return lx.extract(**extract_kwargs)


def save_results(results, output_name="extraction_results.jsonl", output_dir="."):
    """
    Save one or more LangExtract results to a ``.jsonl`` file.

    Parameters
    ----------
    results : AnnotatedDocument or list of AnnotatedDocument
        Result(s) from :func:`extract`.
    output_name : str
        Filename for the saved ``.jsonl`` file.
    output_dir : str
        Directory to write it into (created if missing).

    Returns
    -------
    str: the full path to the saved ``.jsonl`` file.
    """
    lx = _require_langextract()

    if not isinstance(results, (list, tuple)):
        results = [results]

    os.makedirs(output_dir, exist_ok=True)
    lx.io.save_annotated_documents(results, output_name=output_name, output_dir=output_dir)

    path = os.path.join(output_dir, output_name)
    logger.info(f'Saved {len(results)} langextract result(s) to: {path}')
    return path


def visualize(source, output_html=None):
    """
    Generate an interactive, self-contained HTML visualization that
    highlights every extraction in its original context -- the easiest way
    to review LangExtract results.

    Parameters
    ----------
    source : str, AnnotatedDocument, or list of AnnotatedDocument
        Either a path to an existing ``.jsonl`` file (as produced by
        :func:`save_results`), or one/many in-memory results straight from
        :func:`extract` -- these are saved to a temporary ``.jsonl`` file
        automatically before rendering.
    output_html : str, optional
        When given, the HTML is also written to this path so it can be
        opened directly in a browser.

    Returns
    -------
    str: the raw HTML content. In a Jupyter/Colab notebook you can render
    it inline with ``from IPython.display import HTML; HTML(html)``.

    Examples
    --------
    >>> html = visualize(result, output_html="visualization.html")
    >>> # then open visualization.html in any browser, or in Jupyter:
    >>> # from IPython.display import HTML; HTML(html)
    """
    lx = _require_langextract()

    jsonl_path = source
    if not (isinstance(source, str) and source.lower().endswith('.jsonl') and os.path.isfile(source)):
        # Treat `source` as in-memory result(s); persist to a temp jsonl first.
        tmpdir = tempfile.mkdtemp(prefix='llmlight_langextract_')
        jsonl_path = save_results(source, output_name='extraction_results.jsonl', output_dir=tmpdir)

    html_content = lx.visualize(jsonl_path)
    html_str = html_content.data if hasattr(html_content, 'data') else html_content

    if output_html:
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_str)
        logger.info(f'Visualization written to: {output_html}')

    return html_str
