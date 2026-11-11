"""Phase 4: an LLM adjudicates the records Splink is unsure about; deterministic code checks every answer.

select  -> which records go to the model (uncertain band, cardiology by the sources' own fields), never gold
prompt  -> one record and its registry candidates as text, candidates in a shuffled order
verify  -> parse the reply, check its claims against the actual fields, ask once for a repair
run     -> orchestrate, then score Splink alone vs Splink + LLM with the Phase 3 evaluation (the only gold reader)
"""
