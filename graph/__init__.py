"""Phase 5: a lightweight identity graph. Parquet files plus DuckDB views; no server, no graph database, no model calls.

build    -> graph_links.parquet (every candidate decision, with provenance) and graph_entities.parquet (record -> entity),
            from Splink scores and cached adjudication decisions. Reads stripped data only, never gold.
views    -> DuckDB views and two lookups (find entities, entity profile): the tools the question agent will call
evaluate -> cluster metrics against gold, Splink alone vs Splink + LLM (the only gold reader here)
"""
