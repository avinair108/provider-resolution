"""Candidate generators: the Splink model and the exact-match baselines it must beat.

Every generator writes the same table shape, so one evaluation scores them all:
    <name>(a VARCHAR, b VARCHAR, p DOUBLE)  with a < b, one row per scored pair
Reads only the standardized inputs from matching.inputs, never gold.
"""

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, block_on
from splink.backends.duckdb import DuckDBAPI

from matching.inputs import SOURCES

SOURCE_PAIRS = [("nppes", "open_payments"), ("nppes", "part_d"), ("open_payments", "part_d")]

# Blocking chosen by measured recall on Texas: these keep 99.56% of true cross-source pairs in about 10.6M candidates.
# Adding house number + ZIP + first initial doubled the candidates for 0.01 points.
BLOCKING_RULES = [
    block_on("last_name", "first_initial"),
    block_on("last_name_alt", "first_initial"),
    block_on("last_token", "first_initial"),
    block_on("first_name", "zip"),
    block_on("last_name", "zip"),
    "l.first_name = r.last_name AND l.last_name = r.first_name",
]


def exact_baseline(con, name: str, columns: list[str]) -> int:
    """Link records that agree exactly on every column, with p = 1."""
    condition = " AND ".join(f"l.{c} = r.{c}" for c in columns)
    union = " UNION ".join(
        f"SELECT least(l.unique_id, r.unique_id) AS a, greatest(l.unique_id, r.unique_id) AS b, CAST(1.0 AS DOUBLE) AS p "
        f"FROM {x} l JOIN {y} r ON {condition}"
        for x, y in SOURCE_PAIRS
    )
    con.execute(f"CREATE OR REPLACE TABLE {name} AS {union}")
    return con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]


def _first_name() -> cl.CustomComparison:
    return cl.CustomComparison(
        output_column_name="first_name",
        comparison_levels=[
            cll.NullLevel("first_name"),
            cll.ExactMatchLevel("first_name", term_frequency_adjustments=True),
            cll.CustomLevel("first_name_formal_l = first_name_formal_r", "nickname of the same name"),
            cll.CustomLevel("jaro_winkler_similarity(first_name_l, first_name_r) >= 0.92", "Jaro-Winkler >= 0.92"),
            cll.CustomLevel("jaro_winkler_similarity(first_name_l, first_name_r) >= 0.80", "Jaro-Winkler >= 0.80"),
            cll.CustomLevel("first_initial_l = first_initial_r AND (length(first_name_l) = 1 OR length(first_name_r) = 1)",
                            "initial only, consistent"),
            # Swapped names are scored once, in the surname comparison: a second copy here double-counted the
            # evidence and its weight collapsed to zero during training.
            cll.ElseLevel(),
        ],
    )


def _last_name() -> cl.CustomComparison:
    return cl.CustomComparison(
        output_column_name="last_name",
        comparison_levels=[
            cll.NullLevel("last_name"),
            cll.ExactMatchLevel("last_name", term_frequency_adjustments=True),
            cll.CustomLevel("last_name_alt_l = last_name_r OR last_name_l = last_name_alt_r", "matches the registry's other name"),
            cll.CustomLevel("last_token_l = last_token_r", "first word of the surname matches"),
            cll.CustomLevel("jaro_winkler_similarity(last_name_l, last_name_r) >= 0.92", "Jaro-Winkler >= 0.92"),
            cll.CustomLevel("jaro_winkler_similarity(last_name_l, last_name_r) >= 0.80", "Jaro-Winkler >= 0.80"),
            cll.CustomLevel("first_name_l = last_name_r AND last_name_l = first_name_r", "first and last swapped"),
            cll.ElseLevel(),
        ],
    )


def _middle_initial() -> cl.CustomComparison:
    return cl.CustomComparison(
        output_column_name="middle_initial",
        comparison_levels=[
            cll.NullLevel("middle_initial"),
            cll.ExactMatchLevel("middle_initial"),
            cll.ElseLevel(),
        ],
    )


def _location() -> cl.CustomComparison:
    """Practice address, with the registry's mailing address accepted as an alternative."""
    same_zip = "(zip_l = zip_r OR zip_l = alt_zip_r OR alt_zip_l = zip_r)"
    same_building = ("((zip_l = zip_r AND street_number_l = street_number_r) "
                     "OR (zip_l = alt_zip_r AND street_number_l = alt_street_number_r) "
                     "OR (alt_zip_l = zip_r AND alt_street_number_l = street_number_r))")
    return cl.CustomComparison(
        output_column_name="location",
        comparison_levels=[
            cll.NullLevel("zip"),
            cll.CustomLevel(same_building, "same house number and ZIP"),
            cll.CustomLevel(same_zip, "same ZIP"),
            cll.CustomLevel("city_l = city_r", "same city"),
            cll.CustomLevel("left(zip_l, 3) = left(zip_r, 3)", "same 3-digit ZIP area"),
            cll.ElseLevel(),
        ],
    )


def settings() -> SettingsCreator:
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="unique_id",
        comparisons=[_first_name(), _last_name(), _middle_initial(), _location()],
        blocking_rules_to_generate_predictions=BLOCKING_RULES,
        retain_intermediate_calculation_columns=False,
        retain_matching_columns=False,
    )


def splink_model(con, name: str, seed: int, u_max_pairs: float = 1e7) -> dict:
    """Train on the inputs (no labels: random-sampling u, EM for m), score every blocked pair into `name`."""
    db = DuckDBAPI(connection=con)
    frames = [db.register(source, dataset_display_name=source) for source in SOURCES]
    linker = Linker(frames, settings(), log_level=40)
    # Prior: share of pairs that are matches, from rules assumed to find about 70% of them (measured on the
    # exact name + ZIP baseline in Texas; this uses no labels at run time).
    linker.training.estimate_probability_two_random_records_match(
        [block_on("last_name", "first_name", "zip")], recall=0.7)
    linker.training.estimate_u_using_random_sampling(max_pairs=u_max_pairs, seed=seed)
    # Two EM passes, each blocked on what the other one trains.
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("last_name", "first_name"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("zip", "street_number"))
    predictions = linker.inference.predict()
    con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT least(unique_id_l, unique_id_r) AS a, greatest(unique_id_l, unique_id_r) AS b,
               max(match_probability) AS p
        FROM {predictions.physical_name} GROUP BY 1, 2
    """)
    return {"model": linker.misc.save_model_to_json(), "pairs": con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]}
