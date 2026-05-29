"""let's do some front end business. as with the visualizations script, I will probably not be able to give this one as much love as it deserves.
But for now, lets make an interactive narrative dashboard that uses the bits and bobs we have already created: 
our heat map, the graph context text, and the llm answer.

scoping is going to be important here too: One question, one region, one screen. 
THe user interface will be appropriately simple, but I think the key info can be dumped up top/ 
The map shows the city and its surface temperature; 
the middle panel shows exactly which graph nodes drove retrieval;
the right panel shows the cited, LLM-generated answer. They share a single
selection so the three views stay in step rather than being separate demos.

Run with:  uv run streamlit run app/dashboard.py
"""

from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from geokg import boundaries, config, graph, heat, index, rag, retrieval, viz

st.set_page_config(page_title="Swiss Urban Heat: Geo-RAG", layout="wide")


# Cached resources: built once and reused across reruns/interactions.
@st.cache_resource
def _graph():
    return graph.build_graph()


@st.cache_resource
def _client():
    return index.get_client()


@st.cache_data
def _boundaries() -> "pd.DataFrame":
    return boundaries.build_city_boundaries()


@st.cache_data
def _heat() -> pd.DataFrame:
    return heat.build_heat_series()


def _region_label(city: config.City) -> str:
    return f"{city.name} (canton {city.canton})"


def build_map(city_name: str):
    """Folium map: base tiles, the city boundary, and the LST overlay."""
    gdf = _boundaries()
    geom = gdf.loc[city_name, "geometry"]
    minx, miny, maxx, maxy = geom.bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=12, tiles="CartoDB positron")

    overlay = heat.overlay_path(city_name)
    legend = None
    if overlay:
        rgba, bounds, vmin, vmax = viz.lst_to_rgba(overlay)
        folium.raster_layers.ImageOverlay(
            image=rgba, bounds=bounds, opacity=0.6, name="Surface temperature"
        ).add_to(m)
        legend = (vmin, vmax)

    folium.GeoJson(
        gdf.loc[[city_name]].to_json(),
        name="City boundary",
        style_function=lambda _: {"color": "#1f4e79", "weight": 2, "fill": False},
    ).add_to(m)
    return m, legend


def graph_panel(city: config.City, ret: retrieval.Retrieval, summary: dict):
    """Show the graph context: trend numbers, selected documents, datasets."""
    st.subheader("Graph context")

    if summary:
        a, b, c = st.columns(3)
        a.metric("Mean summer LST", f"{summary['mean_c']} °C")
        b.metric("Decadal trend", f"{summary['trend_c_per_decade']:+} °C",
                 help="Least-squares slope over the per-year scenes, scaled to a decade.")
        c.metric("Year-to-year spread", f"±{summary['year_to_year_std_c']} °C")

    st.caption(
        f"Documents reached from **{city.name}** by walking `within` "
        "(city → canton → country):"
    )
    for d in ret.candidate_documents:
        st.markdown(
            f"- [{d['title']}]({d['url']}) · {d['publisher']}  "
            f"`{d['matched_level']}: {d['matched_region']}`"
        )

    ds = ", ".join(
        f"{x['label']}" + (f" ({x['temporal']})" if x["temporal"] else "")
        for x in ret.datasets
    )
    st.caption(f"Datasets covering this region: {ds}")


def answer_panel(question: str, city: config.City, ret: retrieval.Retrieval, summary: dict):
    """Show the grounded answer (if a key is set) and the retrieved evidence."""
    st.subheader("Answer")

    # Either path produces an `Answer`. We never let an API failure crash the page:
    # any error from the LLM call drops to the same templated fallback as the no-key
    # path, with a short notice explaining what happened.
    if rag.is_configured():
        try:
            with st.spinner("Generating grounded answer…"):
                ans = rag.generate_answer(question, _region_label(city), ret, summary)
        except Exception as exc:
            ans = rag.fallback_answer(_region_label(city), ret, summary)
            st.warning(
                f"LLM call failed ({type(exc).__name__}): {exc}. "
                "Showing the templated summary instead."
            )
    else:
        ans = rag.fallback_answer(_region_label(city), ret, summary)
        st.caption(
            "Generated without an LLM (no `OPENROUTER_API_KEY` set): a templated summary "
            "of the retrieved evidence. Set a key for a written narrative."
        )

    st.markdown(ans.answer)
    if ans.data_caveat:
        st.info(ans.data_caveat, icon="⚠")
    cited = [d for d in ret.candidate_documents if d["doc_id"] in ans.sources_used]
    if cited:
        st.caption("Cited sources:")
        for d in cited:
            st.markdown(f"- [{d['title']}]({d['url']})")

    with st.expander(f"Retrieved chunks (graph filter: {'on' if ret.used_graph else 'off'})"):
        for c in ret.chunks:
            st.markdown(f"**{c['doc_id']}** · score {c['score']}")
            st.write(c["text"])


def main():
    st.title("How has urban heat changed in this region over the last decade?")
    """st.write(
        "Surface temperature, Swiss boundary data, and city heat-plans, "
        "linked through a knowledge graph and answered with cited sources."
    )"""

    with st.sidebar:
        st.header("Query")
        city_name = st.selectbox("City", [c.name for c in config.CITIES])
        city = config.city_by_name(city_name)
        question = st.text_area(
            "Question",
            value=f"How has urban heat changed in {city_name} over the last decade, "
                  "and what is being done about it?",
            height=110,
        )
        top_k = st.slider("Chunks to retrieve", 2, 8, 4)
        use_graph = st.checkbox(
            "Use graph to filter retrieval", value=True,
            help="Off = plain vector search over all documents, so you can see "
                 "the graph's effect on which sources are in scope.",
        )
        go = st.button("Run", type="primary")

    heat_df = _heat()
    summary = heat.heat_summary(heat_df, city_name)
    ret = retrieval.retrieve(question, city_name, top_k=top_k, use_graph=use_graph,
                             store=_graph(), client=_client())

    left, right = st.columns([3, 2])
    with left:
        fmap, legend = build_map(city_name)
        st_folium(fmap, height=460, use_container_width=True, returned_objects=[])
        if legend:
            st.caption(f"Overlay: summer land surface temperature, "
                       f"~{legend[0]} to {legend[1]} °C (Landsat, latest clear scene).")
        series = (heat_df[heat_df["city"] == city_name]
                  .set_index("year")["mean_lst_c"])
        st.caption("Mean summer surface temperature by year (°C):")
        st.line_chart(series)

    with right:
        graph_panel(city, ret, summary)

    st.divider()
    # The answer is the one place we call the LLM, so gate it behind the button.
    if go:
        answer_panel(question, city, ret, summary)
    else:
        st.caption("Press **Run** in the sidebar to generate the cited answer.")


if __name__ == "__main__":
    main()
