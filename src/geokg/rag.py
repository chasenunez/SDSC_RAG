""" now we leave the firm ground of geospatial interpolation and venture into the squishy marshlands of answer generation.
The LLM is only ever supposed to see the chunks the graph-filtered retrieval selected plus the heat statistics for the region. 
It is asked to list the documents it used by id, and we drop any id it returns that was not actually supplied, so a citation can
never point at a source the model did not see. 

Generation needs an OpenRouter key; everything upstream (graph, retrieval) runs without one.

This code is inspired by some work i did as part of a course on creating an LLM from scratch on boot.dev. It has been substantially chnaged, but 
it has also been adapted to use pydantic_ai as suggested in the prompt. 
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from . import config, documents
from .retrieval import Retrieval

_SYSTEM_PROMPT = (
    "You are a geospatial analyst answering questions about urban heat in Swiss "
    "cities. Use only the numbered SOURCES and the HEAT STATISTICS provided. Do "
    "not add facts, figures, or sources from outside them. Cite the sources you "
    "rely on by their id. The heat statistics come from individual clear-sky "
    "Landsat scenes, one per year, so they are weather-noisy; if the year-to-year "
    "spread is comparable to the trend, say the decadal trend is not statistically "
    "clear rather than overstating it. Be concise and specific."
)


class Answer(BaseModel):
    """Structured, grounded answer returned to the dashboard."""

    answer: str = Field(description="Concise narrative answer for the user.")
    sources_used: list[str] = Field(
        default_factory=list, description="ids of the SOURCES actually relied on."
    )
    data_caveat: str = Field(
        description="One line on the limits/uncertainty of the heat data used."
    )


def is_configured() -> bool:
    """True if an OpenRouter key is available for the generation step."""
    config.load_env()
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _agent() -> Agent:
    provider = OpenAIProvider(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    model = OpenAIChatModel(config.model_name(), provider=provider)
    return Agent(model, output_type=Answer, system_prompt=_SYSTEM_PROMPT)


def _build_prompt(question: str, region_label: str, ret: Retrieval, heat: dict) -> str:
    """Assemble the user prompt from the region, heat numbers, and source chunks."""
    sources = "\n".join(
        f"[{c['doc_id']}] {c['title']} ({c['url']})\n{c['text']}" for c in ret.chunks
    )
    heat_lines = (
        f"city={heat.get('city')}, years {heat.get('first_year')}-{heat.get('last_year')} "
        f"({heat.get('n_years')} scenes), mean LST {heat.get('mean_c')} C, "
        f"trend {heat.get('trend_c_per_decade')} C/decade, "
        f"year-to-year std {heat.get('year_to_year_std_c')} C, "
        f"warmest {heat.get('warmest')}, coolest {heat.get('coolest')}"
        if heat else "no usable heat data for this region"
    )
    return (
        f"QUESTION: {question}\n\n"
        f"REGION: {region_label}\n\n"
        f"HEAT STATISTICS (Landsat surface temperature, summer scenes):\n{heat_lines}\n\n"
        f"SOURCES:\n{sources}"
    )


def generate_answer(question: str, region_label: str, ret: Retrieval, heat: dict) -> Answer:
    """Generate a grounded, cited answer. Raises if no API key is configured."""
    if not is_configured():
        raise RuntimeError("OPENROUTER_API_KEY is not set; cannot generate an answer.")

    result = _agent().run_sync(_build_prompt(question, region_label, ret, heat))
    answer = result.output

    # Grounding guard: never let a citation reference a source we did not supply.
    supplied = {c["doc_id"] for c in ret.chunks}
    answer.sources_used = [s for s in answer.sources_used if s in supplied]
    return answer


_LST_CAVEAT = (
    "Surface temperature here is one clear-sky summer scene per year, so each value "
    "carries that day's weather; read the decadal trend as indicative, not definitive."
)


def fallback_answer(region_label: str, ret: Retrieval, heat: dict) -> Answer:
    """Deterministic, source-grounded summary for when no LLM key is set.

    This is not the LLM narrative; it stitches the heat numbers and the lead
    sentence of each retrieved chunk into a cited paragraph, so the dashboard's
    narrative panel still shows something real and traceable without a key.
    """
    if heat:
        trend = heat["trend_c_per_decade"]
        lead = (
            f"Across {heat['n_years']} summer scenes ({heat['first_year']}-{heat['last_year']}), "
            f"mean land surface temperature in {region_label} averaged {heat['mean_c']} C, with a "
            f"least-squares trend of {trend:+} C per decade and a year-to-year spread of "
            f"{heat['year_to_year_std_c']} C. The spread is comparable to the trend, so the "
            f"decadal signal is not statistically clear from this sample."
        )
    else:
        lead = f"No usable cloud-free summer scenes were found for {region_label}."

    # One bullet per unique retrieved document, using its clean opening sentence
    # (chunks can start mid-sentence because of overlap, so read from the source).
    docs_by_id = {d["id"]: d for d in documents.load_documents()}
    points, used = [], []
    for chunk in ret.chunks:
        doc_id = chunk["doc_id"]
        if doc_id in used:
            continue
        used.append(doc_id)
        record = docs_by_id.get(doc_id)
        opening = record["content"].split(". ")[0].strip() if record else chunk["title"]
        points.append(f"- {chunk['title']}: {opening}. [{doc_id}]")
        if len(used) >= 3:
            break

    body = lead
    if points:
        body += "\n\nFrom the documents the graph linked to this region:\n" + "\n".join(points)
    return Answer(answer=body, sources_used=used, data_caveat=_LST_CAVEAT)
