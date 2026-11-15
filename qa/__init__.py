"""The question-answering layer over the identity graph.

questions -> a seeded question set with gold answers computed from the hidden-NPI truth (not from the graph), so
             linkage mistakes in the graph show up as wrong answers. Written to data/<split>/<state>/questions/,
             which is git-ignored: questions name real providers and payments.
"""
