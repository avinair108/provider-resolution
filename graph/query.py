"""Look things up in a built graph from the terminal. Local only; prints real names, so keep output out of the repo.

  python -m graph.query --last SMITH --first JOHN --city AUSTIN
  python -m graph.query --entity E1a2b3c4d5e6f
"""

import argparse
import json
from pathlib import Path

import duckdb

from graph import views

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--graph", default="graph", help="graph dir name under data/<split>/<state>/")
    parser.add_argument("--last")
    parser.add_argument("--first")
    parser.add_argument("--city")
    parser.add_argument("--entity")
    args = parser.parse_args()
    data = ROOT / "data" / args.split / args.state
    con = duckdb.connect()
    views.register(con, data / args.graph, data / "stripped")
    if args.entity:
        result = views.entity_profile(con, args.entity)
    elif args.last:
        result = views.find_entities(con, args.last, args.first, args.city)
    else:
        raise SystemExit("pass --last (with optional --first, --city) or --entity")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
