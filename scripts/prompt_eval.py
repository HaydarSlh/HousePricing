"""Evaluate extraction prompt v1 vs v2 on test queries.

Usage:
    python -m scripts.prompt_eval
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.stage1_extractor import FeatureExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

TEST_QUERIES = [
    # 1. Full description — all 12 features extractable
    (
        "A 2005-built home in StoneBr, 2400 sqft living area, excellent exterior "
        "and kitchen, good basement quality with 1200 sqft basement, 2 full baths, "
        "a 600 sqft garage, good fireplace, and 200 sqft of stone veneer. "
        "Overall quality is 8 out of 10."
    ),
    # 2. Sparse — very few features
    (
        "A small house in the Old Town neighborhood."
    ),
    # 3. Ambiguous / conversational
    (
        "Looking for a newer home, something built in the 2000s, near the college. "
        "I want at least 2 bathrooms and a big open floor plan, maybe 2000 square "
        "feet or more. The kitchen should be modern and nice."
    ),
    # 4. Absence signals
    (
        "Ranch-style home, no basement, no garage, no fireplace. Built in 1965 in "
        "North Ames, about 1100 sqft, 1 bathroom, average exterior, no stone veneer. "
        "Quality around 5."
    ),
]


def run_evaluation():
    for version in ("v1", "v2"):
        extractor = FeatureExtractor(prompt_version=version)

        print(f"\n{'=' * 70}")
        print(f"  PROMPT VERSION: {version}")
        print(f"{'=' * 70}")

        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n--- Query {i} ---")
            print(f"Input: {query[:80]}...")

            try:
                result = extractor.extract(query)

                print(f"Complete   : {result.is_complete}")
                print(f"Extracted  : {result.extracted_fields}")
                print(f"Missing    : {result.missing_fields}")
                print(f"Completeness: {result.completeness_ratio:.0%}")

                if result.features:
                    vals = result.features.model_dump()
                    print(f"Values     : {json.dumps(vals, indent=2)}")

                if result.missing_details:
                    print("Missing details:")
                    for d in result.missing_details:
                        desc = d.get("description", "")
                        rng = d.get("valid_range", "")
                        opts = d.get("valid_options", "")
                        print(f"  - {d['field']}: {desc}")
                        if rng:
                            print(f"    Range: {rng}")
                        if opts:
                            print(f"    Options: {opts}")

            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")

            print()


if __name__ == "__main__":
    run_evaluation()
